package RAGApp;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for loading documents from file and splitting them into chunks.
 */
public class DocumentLoader {

    /**
     * Loads a document from the given file path and splits it into chunks.
     *
     * @param filePath  Path to the document file
     * @param chunkSize The size of each chunk (in characters)
     * @return A list of text chunks
     * @throws IOException If file reading fails
     */
    public static List<String> loadDocumentChunks(String filePath, int chunkSize) throws IOException {
        String content = Files.readString(Paths.get(filePath), StandardCharsets.UTF_8);
        return splitIntoChunks(content, chunkSize);
    }

    /**
     * Splits the given text into fixed-size chunks.
     *
     * @param text      The text to split
     * @param chunkSize The size of each chunk
     * @return A list of chunks
     */
    private static List<String> splitIntoChunks(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        int length = text.length();

        for (int i = 0; i < length; i += chunkSize) {
            int end = Math.min(length, i + chunkSize);
            chunks.add(text.substring(i, end));
        }

        System.out.println("Loaded " + chunks.size() + " chunks.");
        return chunks;
    }
}
