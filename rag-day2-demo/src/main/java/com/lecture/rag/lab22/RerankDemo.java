package com.lecture.rag.lab22;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Lab 2.2 — Rerank: 넓게 검색(top-20) 후 LLM으로 재채점해서 top-5만 채택.
 * 순수 벡터 검색 top-5와 rerank 결과 top-5를 나란히 비교한다.
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab22
 */
@Component
@Profile("lab22")
public class RerankDemo implements CommandLineRunner {

    private final EmbeddingModel embeddingModel;
    private final ChatModel chatModel;

    public RerankDemo(EmbeddingModel embeddingModel, ChatModel chatModel) {
        this.embeddingModel = embeddingModel;
        this.chatModel = chatModel;
    }

    @Override
    public void run(String... args) {
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader("classpath:/docs/agentic-rag-survey.pdf");
        List<Document> documents = pdfReader.get();

        TokenTextSplitter splitter = TokenTextSplitter.builder().withChunkSize(300).build();
        List<Document> chunks = splitter.apply(documents);

        VectorStore vectorStore = SimpleVectorStore.builder(embeddingModel).build();
        vectorStore.add(chunks);
        System.out.println("총 청크 수: " + chunks.size());
        System.out.println();

        String query = "What are the main failure modes of agentic RAG systems?";

        // 1) 순수 벡터 검색 top-5
        List<Document> plainTop5 = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(5).build());

        System.out.println("=== 1. 순수 벡터 검색 top-5 ===");
        printPreview(plainTop5);
        System.out.println();

        // 2) 넓게 검색(top-20) 후 LLM rerank로 top-5 재선정
        List<Document> wideCandidates = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(20).build());

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 숫자만 답하세요.")
                .build();
        LlmReranker reranker = new LlmReranker(chatClient);

        System.out.println("=== 2. 넓게 검색(top-20) 후보 각각의 LLM 채점 ===");
        List<LlmReranker.Scored> scored = reranker.scoreAll(query, wideCandidates);
        scored.stream()
                .sorted((a, b) -> Integer.compare(b.score(), a.score()))
                .forEach(s -> System.out.printf("점수 %2d | %s...%n", s.score(), preview(s.doc(), 70)));
        System.out.println();

        List<Document> reranked = scored.stream()
                .sorted((a, b) -> Integer.compare(b.score(), a.score()))
                .limit(5)
                .map(LlmReranker.Scored::doc)
                .toList();

        System.out.println("=== 3. Rerank 최종 top-5 ===");
        printPreview(reranked);
        System.out.println();

        // 4) 두 결과가 실제로 다른지 확인
        long overlap = plainTop5.stream().filter(reranked::contains).count();
        System.out.println("순수 벡터 top-5와 rerank top-5의 겹치는 문서 수: " + overlap + " / 5");
    }

    private void printPreview(List<Document> docs) {
        for (Document doc : docs) {
            System.out.println("- " + preview(doc, 100) + "...");
        }
    }

    private String preview(Document doc, int len) {
        String text = doc.getText().replaceAll("\\s+", " ").trim();
        return text.substring(0, Math.min(len, text.length()));
    }
}
