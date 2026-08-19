package com.lecture.rag.lab21;

import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Lab 2.1 — Day1의 SimpleVectorStore를 PgVectorStore로 교체.
 * VectorStore는 인터페이스라서 add()/similaritySearch() 호출부는 Day1과 완전히 동일 —
 * 바뀐 건 Bean 구현체(PgVectorStore가 자동 주입됨)뿐이라는 게 이 랩의 핵심 포인트.
 *
 * 실행: 1) docker compose up -d  (PGVector 컨테이너 기동)
 *       2) ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab21
 *
 * 영속성 비교 실습: 이 프로필로 한 번 인덱싱한 뒤 앱을 완전히 종료했다가 다시 실행해보면,
 * "재인덱싱하지 않았는데도" 검색이 바로 되는 걸 확인할 수 있다 (SimpleVectorStore였다면
 * 재시작 시 데이터가 전부 사라져서 매번 재인덱싱해야 했음).
 */
@Component
@Profile("lab21")
public class PgVectorIndexingDemo implements CommandLineRunner {

    private final VectorStore vectorStore; // PgVectorStore가 자동 주입됨 (application.yml 설정 기반)

    public PgVectorIndexingDemo(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    @Override
    public void run(String... args) {
        long existingCount = countExisting();
        System.out.println("=== 시작 시점 PGVector에 이미 저장된 문서 수: " + existingCount + " ===");

        if (existingCount == 0) {
            System.out.println("=== 처음 실행이라 인덱싱을 진행합니다 ===");
            index();
        } else {
            System.out.println("=== 이미 인덱싱된 데이터가 있습니다 — 재인덱싱 없이 바로 검색합니다 ===");
            System.out.println("(SimpleVectorStore였다면 재시작 시 이 데이터가 전부 사라졌을 것 — 이게 영속성의 차이)");
        }
        System.out.println();

        String query = "환불은 며칠 안에 가능해?";
        List<Document> results = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(3).build());

        System.out.println("=== 검색 결과 (질문: \"" + query + "\") ===");
        for (Document doc : results) {
            System.out.println("- " + doc.getText().replaceAll("\\s+", " "));
        }
    }

    private void index() {
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader("classpath:/docs/manual.pdf");
        List<Document> documents = pdfReader.get();

        TokenTextSplitter splitter = TokenTextSplitter.builder()
                .withChunkSize(200)
                .build();
        List<Document> chunks = splitter.apply(documents);

        vectorStore.add(chunks);
        System.out.println("생성된 청크 수: " + chunks.size());
    }

    private long countExisting() {
        // 빈 질의로 top-1000 검색해서 기존 저장된 문서 수를 어림잡음 (PgVectorStore엔 count() API가 없음)
        List<Document> all = vectorStore.similaritySearch(
                SearchRequest.builder().query("환불").topK(1000).similarityThresholdAll().build());
        return all.size();
    }
}
