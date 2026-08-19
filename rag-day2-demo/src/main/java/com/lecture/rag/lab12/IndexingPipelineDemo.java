package com.lecture.rag.lab12;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Lab 1.2 — 문서 로딩 -> 청킹 -> VectorStore 인덱싱 -> 검색
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab12
 * 실습 문서: 실제 arXiv 논문 (2026-03, arXiv:2603.07379, 25페이지)
 */
@Component
@Profile("lab12")
public class IndexingPipelineDemo implements CommandLineRunner {

    private static final String DOCUMENT_PATH = "classpath:/docs/agentic-rag-survey.pdf";

    private final EmbeddingModel embeddingModel;

    public IndexingPipelineDemo(EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    @Override
    public void run(String... args) {
        // 1) 문서 로드
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(DOCUMENT_PATH);
        List<Document> documents = pdfReader.get();
        System.out.println("=== 1. 문서 로드 ===");
        System.out.println("로드된 페이지(Document) 수: " + documents.size());
        System.out.println();

        // 2) 청킹
        TokenTextSplitter splitter = TokenTextSplitter.builder()
                .withChunkSize(500)
                .build();
        List<Document> chunks = splitter.apply(documents);
        System.out.println("=== 2. 청킹 결과 ===");
        System.out.println("생성된 청크 수: " + chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            String preview = chunks.get(i).getText().replaceAll("\\s+", " ");
            preview = preview.substring(0, Math.min(60, preview.length()));
            System.out.printf("  [%d] %s...%n", i, preview);
        }
        System.out.println();

        // 3) VectorStore 구성 + 저장 (Day1은 인메모리 SimpleVectorStore, Day2에서 PGVector로 교체)
        VectorStore vectorStore = SimpleVectorStore.builder(embeddingModel).build();
        vectorStore.add(chunks);
        System.out.println("=== 3. VectorStore 저장 완료 (SimpleVectorStore, 인메모리) ===");
        System.out.println();

        // 4) 검색
        String query = "What is agentic RAG?";
        List<Document> results = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(3).build());

        System.out.println("=== 4. 검색 결과 (질문: \"" + query + "\") ===");
        for (Document doc : results) {
            System.out.println("- " + doc.getText().replaceAll("\\s+", " "));
        }
    }
}
