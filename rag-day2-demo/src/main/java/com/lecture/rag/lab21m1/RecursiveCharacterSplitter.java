package com.lecture.rag.lab21m1;

import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;

/**
 * M2.1 — Recursive Character Splitting.
 * "가능하면 자연스러운 경계에서 자르되, 그게 너무 크면 한 단계 더 작은 경계로 내려가서 다시 시도한다."
 * 우선순위: 문단(\n\n) → 문장(. / 다.) → 단어(공백)
 * Spring AI TokenTextSplitter에는 이 기능이 없어 직접 구현.
 */
public class RecursiveCharacterSplitter {

    private final int maxChunkChars;
    // PDF에서 추출한 텍스트의 줄바꿈은 "시각적 줄바꿈"(word-wrap)일 뿐 의미 단락 경계가 아니라서
    // \n을 우선순위 상단에 두면 오히려 문장을 잘게 쪼갠다 — 문장부호(". ")를 최우선으로 둔다.
    private static final String[] SEPARATORS = {". ", "다. ", "\n", " "};

    public RecursiveCharacterSplitter(int maxChunkChars) {
        this.maxChunkChars = maxChunkChars;
    }

    public List<Document> split(Document doc) {
        List<String> pieces = splitRecursive(doc.getText(), 0);
        return pieces.stream().map(Document::new).toList();
    }

    private List<String> splitRecursive(String text, int separatorIndex) {
        List<String> result = new ArrayList<>();
        if (text.length() <= maxChunkChars || separatorIndex >= SEPARATORS.length) {
            if (!text.isBlank()) result.add(text.trim());
            return result;
        }
        String sep = SEPARATORS[separatorIndex];
        String[] parts = text.split(java.util.regex.Pattern.quote(sep));

        StringBuilder current = new StringBuilder();
        for (String part : parts) {
            String candidate = current.isEmpty() ? part : current + sep + part;
            if (candidate.length() > maxChunkChars) {
                if (!current.isEmpty()) {
                    flush(result, current.toString(), separatorIndex);
                    current = new StringBuilder(part);
                } else {
                    flush(result, part, separatorIndex);
                }
            } else {
                current = new StringBuilder(candidate);
            }
        }
        if (!current.isEmpty()) {
            flush(result, current.toString(), separatorIndex);
        }
        return result;
    }

    // current가 이미 maxChunkChars 이내면 그대로 채택하고, 넘을 때만 다음 단계 구분자로 재귀 분할
    private void flush(List<String> result, String text, int separatorIndex) {
        if (text.length() <= maxChunkChars) {
            if (!text.isBlank()) result.add(text.trim());
        } else {
            result.addAll(splitRecursive(text, separatorIndex + 1));
        }
    }
}
