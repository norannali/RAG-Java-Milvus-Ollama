package RAGApp;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for splitting long text into smaller fixed-size chunks.
 */
public class TextSplitter {

    /**
     * Splits the input text into chunks of the given size.
     * This is a naive splitter that does not consider word boundaries.
     *
     * @param text      The text to split
     * @param chunkSize The number of characters per chunk
     * @return A list of text chunks
     */
    public static List<String> split(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        int length = text.length();
        for (int i = 0; i < length; i += chunkSize) {
            int end = Math.min(length, i + chunkSize);
            chunks.add(text.substring(i, end));
        }
        return chunks;
    }
}
