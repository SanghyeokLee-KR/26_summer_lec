package com.lecture.rag.lab21m1;

import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;

/**
 * M2.1 — Sliding Window 청킹. chunk size만큼 창을 만들고, stride(< chunk size)만큼만 이동해서
 * 청크끼리 서로 겹치게 만든다. Spring AI TokenTextSplitter는 overlap 파라미터가 없어 직접 구현.
 */
public class SlidingWindowSplitter {

    private final int windowChars;
    private final int strideChars;

    public SlidingWindowSplitter(int windowChars, int strideChars) {
        this.windowChars = windowChars;
        this.strideChars = strideChars;
    }

    public List<Document> split(Document doc) {
        String text = doc.getText();
        List<Document> chunks = new ArrayList<>();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + windowChars, text.length());
            chunks.add(new Document(text.substring(start, end).trim()));
            if (end == text.length()) break;
            start += strideChars;
        }
        return chunks;
    }
}
