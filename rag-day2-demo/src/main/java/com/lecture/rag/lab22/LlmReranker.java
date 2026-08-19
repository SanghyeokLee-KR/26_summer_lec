package com.lecture.rag.lab22;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;

import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Rerank = 검색으로 넓게 받아온 후보를 LLM이 다시 채점해서 진짜 관련 있는 것만 상위로 올리는 것.
 * 진짜 Cross-encoder(BAAI/bge-reranker 등)는 별도 인프라가 필요해 수업 범위 밖 —
 * 이미 갖고 있는 LLM을 채점기로 재활용하는 "LLM 기반 재정렬"로 대체. 개념은 동일하다:
 * 넓게 검색(wide retrieval) → 정밀 재점수(rerank) → 상위 N개만 채택.
 */
public class LlmReranker {

    private static final Pattern FIRST_NUMBER = Pattern.compile("\\d+");

    private final ChatClient chatClient;

    public LlmReranker(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    public record Scored(Document doc, int score) {}

    public List<Scored> scoreAll(String query, List<Document> candidates) {
        return candidates.stream()
                .map(doc -> new Scored(doc, scoreOne(query, doc)))
                .toList();
    }

    public List<Document> rerank(String query, List<Document> candidates, int topN) {
        return scoreAll(query, candidates).stream()
                .sorted(Comparator.comparingInt(Scored::score).reversed())
                .limit(topN)
                .map(Scored::doc)
                .toList();
    }

    private int scoreOne(String query, Document doc) {
        String prompt = """
                질문: %s
                문서: %s
                이 문서가 질문에 답하는 데 얼마나 관련 있는지 0~10 사이 숫자 하나만 답하세요. 설명은 하지 마세요.
                """.formatted(query, doc.getText());

        String response = chatClient.prompt().user(prompt).call().content().trim();
        Matcher matcher = FIRST_NUMBER.matcher(response);
        // LLM이 "설명 없이 숫자만" 지시를 안 지키고 군더더기를 붙이는 경우가 있어 첫 숫자만 뽑아 방어
        if (matcher.find()) {
            return Math.min(10, Integer.parseInt(matcher.group()));
        }
        return 0;
    }
}
