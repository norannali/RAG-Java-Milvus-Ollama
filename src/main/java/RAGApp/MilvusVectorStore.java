package RAGApp;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.*;
import io.milvus.param.*;
import io.milvus.param.collection.*;
import io.milvus.param.dml.*;
import io.milvus.param.highlevel.collection.ListCollectionsParam;
import io.milvus.param.index.*;
import io.milvus.response.QueryResultsWrapper;
import io.milvus.response.SearchResultsWrapper;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class MilvusVectorStore implements AutoCloseable {
    private final MilvusServiceClient milvusClient;
    private final String collectionName;
    private final int dimension;
    private long currentId = 1;

    private final List<Long> pendingIds = new ArrayList<>();
    private final List<List<Float>> pendingEmbeddings = new ArrayList<>();
    private final List<String> pendingTexts = new ArrayList<>();
    private static final int BATCH_SIZE = 50;
    private long lastFlushTime = 0;
    private static final long FLUSH_INTERVAL_MS = 12000;

    public MilvusVectorStore(int dimension, String host, int port, String collectionName) {
        this.dimension = dimension;
        this.collectionName = collectionName;
        this.milvusClient = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost(host)
                        .withPort(port)
                        .withConnectTimeout(10, TimeUnit.SECONDS)
                        .build());
        createCollectionIfNotExists();
    }

    private void createCollectionIfNotExists() {
        try {
            R<Boolean> hasCollection = milvusClient.hasCollection(
                    HasCollectionParam.newBuilder()
                            .withCollectionName(collectionName)
                            .build());

            if (!hasCollection.getData()) {
                System.out.println("🔨 Creating new collection: " + collectionName);

                FieldType idField = FieldType.newBuilder()
                        .withName("id")
                        .withDataType(DataType.Int64)
                        .withPrimaryKey(true)
                        .withAutoID(false)
                        .build();

                FieldType vectorField = FieldType.newBuilder()
                        .withName("embedding")
                        .withDataType(DataType.FloatVector)
                        .withDimension(dimension)
                        .build();

                FieldType textField = FieldType.newBuilder()
                        .withName("text")
                        .withDataType(DataType.VarChar)
                        .withMaxLength(65535)
                        .build();

                R<RpcStatus> createResponse = milvusClient.createCollection(
                        CreateCollectionParam.newBuilder()
                                .withCollectionName(collectionName)
                                .withDescription("RAG collection")
                                .withShardsNum(2)
                                .addFieldType(idField)
                                .addFieldType(vectorField)
                                .addFieldType(textField)
                                .build());

                if (createResponse.getStatus()!=0) {
                    throw new RuntimeException("Collection creation failed: " + createResponse.getMessage());
                }

                R<RpcStatus> indexResponse = milvusClient.createIndex(
                        CreateIndexParam.newBuilder()
                                .withCollectionName(collectionName)
                                .withFieldName("embedding")
                                .withIndexType(IndexType.IVF_FLAT)
                                .withMetricType(MetricType.L2)
                                .withExtraParam("{\"nlist\":128}")
                                .build());

                if (indexResponse.getStatus()!=0) {
                    throw new RuntimeException("Index creation failed: " + indexResponse.getMessage());
                }

                R<RpcStatus> loadResponse = milvusClient.loadCollection(
                        LoadCollectionParam.newBuilder()
                                .withCollectionName(collectionName)
                                .build());

                if (loadResponse.getStatus()!=0) {
                    throw new RuntimeException("Collection loading failed: " + loadResponse.getMessage());
                }
            } else {
                System.out.println("✅ Collection already exists: " + collectionName);

                R<DescribeCollectionResponse> describeResponse = milvusClient.describeCollection(
                        DescribeCollectionParam.newBuilder()
                                .withCollectionName(collectionName)
                                .build());

                List<FieldSchema> fields = describeResponse.getData().getSchema().getFieldsList();
                Set<String> fieldNames = new HashSet<>();
                for (FieldSchema field : fields) {
                    fieldNames.add(field.getName());
                }

                List<String> missingFields = new ArrayList<>();
                for (String required : Arrays.asList("id", "embedding", "text")) {
                    if (!fieldNames.contains(required)) {
                        missingFields.add(required);
                    }
                }

                if (!missingFields.isEmpty()) {
                    throw new RuntimeException("Collection exists but missing required fields: " + missingFields);
                }

                R<RpcStatus> loadResponse = milvusClient.loadCollection(
                        LoadCollectionParam.newBuilder()
                                .withCollectionName(collectionName)
                                .build());

                if (loadResponse.getStatus()!=0) {
                    System.out.println("⚠️ Collection load warning: " + loadResponse.getMessage());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize Milvus collection", e);
        }
    }

    public boolean isConnected() {
        try {
            return milvusClient != null &&
                    milvusClient.listCollections(ListCollectionsParam.newBuilder().build()).getStatus()!=0;
        } catch (Exception e) {
            return false;
        }
    }

    public void index(String text, float[] embeddingArray) {
        List<Float> embedding = new ArrayList<>();
        for (float f : embeddingArray) {
            embedding.add(f);
        }

        pendingIds.add(currentId);
        pendingEmbeddings.add(embedding);
        pendingTexts.add(text);

        if (pendingIds.size() >= BATCH_SIZE ||
                System.currentTimeMillis() - lastFlushTime > FLUSH_INTERVAL_MS) {
            flush();
        }

        currentId++;
    }

    private void flush() {
        if (pendingIds.isEmpty()) return;

        List<InsertParam.Field> fields = new ArrayList<>();
        fields.add(new InsertParam.Field("id", pendingIds));
        fields.add(new InsertParam.Field("embedding", pendingEmbeddings));
        fields.add(new InsertParam.Field("text", pendingTexts));

        R<MutationResult> insertResult = milvusClient.insert(
                InsertParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withFields(fields)
                        .build());

        if (insertResult.getStatus()!=0) {
            throw new RuntimeException("Insert failed: " + insertResult.getMessage());
        }

        milvusClient.flush(FlushParam.newBuilder()
                .withCollectionNames(Collections.singletonList(collectionName))
                .withSyncFlush(true)
                .build());

        pendingIds.clear();
        pendingEmbeddings.clear();
        pendingTexts.clear();
        lastFlushTime = System.currentTimeMillis();
    }

    public Map<Long, String> search(float[] queryEmbedding, int topK) {
        List<List<Float>> vectorsToSearch = new ArrayList<>();
        List<Float> vector = new ArrayList<>();
        for (float f : queryEmbedding) vector.add(f);
        vectorsToSearch.add(vector);

        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(MetricType.L2)
                .withOutFields(Arrays.asList("id", "text"))  // 🟢 نطلب النص كمان
                .withTopK(topK)
                .withVectors(vectorsToSearch)
                .withVectorFieldName("embedding")
                .withParams("{\"nprobe\":10}")
                .build();

        R<SearchResults> searchResults = milvusClient.search(searchParam);
        if (searchResults.getStatus()!=0) {
            throw new RuntimeException("Search failed: " + searchResults.getMessage());
        }

        SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResults.getData().getResults());
        List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(0);

        Map<Long, String> results = new LinkedHashMap<>();
        for (int i = 0; i < scores.size(); i++) {
            Long id = scores.get(i).getLongID();
            String text = wrapper.getFieldData("text", 0).toString();
            results.put(id, text);
        }

        return results;
    }


    @Override
    public void close() {
        if (milvusClient != null) {
            milvusClient.close();
        }
    }
}
