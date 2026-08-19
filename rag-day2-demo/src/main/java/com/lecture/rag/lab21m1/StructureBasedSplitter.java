package com.lecture.rag.lab21m1;

import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * M2.1 — 문서 구조 활용 청킹. 문서에 이미 있는 구조적 경계(마크다운 헤더, "제N조" 같은 조항 번호 등)를
 * 그대로 청크 경계로 사용한다. 표는 통째로 유지된다는 원칙과 같은 맥락 — 구조 단위를 절대 쪼개지 않음.
 */
public class StructureBasedSplitter {

    private final Pattern boundaryPattern;

    public StructureBasedSplitter(Pattern boundaryPattern) {
        this.boundaryPattern = boundaryPattern;
    }

    public static StructureBasedSplitter forKoreanArticles() {
        return new StructureBasedSplitter(Pattern.compile("(제\\s*\\d+\\s*조)"));
    }

    public static StructureBasedSplitter forMarkdownHeaders() {
        return new StructureBasedSplitter(Pattern.compile("(?m)^(#{1,6}\\s+.+)$"));
    }

    public List<Document> split(Document doc) {
        String text = doc.getText();
        Matcher matcher = boundaryPattern.matcher(text);

        List<Integer> boundaries = new ArrayList<>();
        while (matcher.find()) {
            boundaries.add(matcher.start());
        }

        List<Document> chunks = new ArrayList<>();
        if (boundaries.isEmpty()) {
            chunks.add(new Document(text.trim()));
            return chunks;
        }
        if (boundaries.get(0) > 0) {
            chunks.add(new Document(text.substring(0, boundaries.get(0)).trim()));
        }
        for (int i = 0; i < boundaries.size(); i++) {
            int start = boundaries.get(i);
            int end = (i + 1 < boundaries.size()) ? boundaries.get(i + 1) : text.length();
            String piece = text.substring(start, end).trim();
            if (!piece.isEmpty()) chunks.add(new Document(piece));
        }
        return chunks;
    }
}
