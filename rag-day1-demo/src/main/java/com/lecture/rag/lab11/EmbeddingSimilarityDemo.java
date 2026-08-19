package com.lecture.rag.lab11;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Lab 1.1 — 프로젝트 셋업 + 임베딩 유사도 실습
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab11
 */
@Component
@Profile("lab11")
public class EmbeddingSimilarityDemo implements CommandLineRunner {

    private final EmbeddingModel embeddingModel;

    public EmbeddingSimilarityDemo(EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    @Override
    public void run(String... args) {
        // 1) 임베딩 호출 + 차원 확인
        float[] vector = embeddingModel.embed("커피메이커 물탱크 용량은 얼마인가요?");
        System.out.println("=== 임베딩 차원 확인 ===");
        System.out.println("벡터 차원 수: " + vector.length);
        System.out.println("앞부분 5개 값: " + previewFirstN(vector, 5));
        System.out.println();

        // 2) 문장 비교 — 의미 비슷한 문장 vs 무관한 문장
        String base = "이 커피메이커의 물탱크 용량이 궁금해요";
        String similar = "MCM-200 물통에는 물이 얼마나 들어가나요?";      // 의미는 비슷, 표현은 다름
        String unrelated = "오늘 저녁 메뉴로 뭐가 좋을까요?";              // 완전히 무관

        float[] baseVec = embeddingModel.embed(base);
        float[] similarVec = embeddingModel.embed(similar);
        float[] unrelatedVec = embeddingModel.embed(unrelated);

        System.out.println("=== 코사인 유사도 비교 ===");
        System.out.printf("[기준]      %s%n", base);
        System.out.printf("[의미 유사] %s  -> 유사도 %.4f%n", similar, cosineSimilarity(baseVec, similarVec));
        System.out.printf("[의미 무관] %s  -> 유사도 %.4f%n", unrelated, cosineSimilarity(baseVec, unrelatedVec));
        System.out.println();
        System.out.println("기대 결과: 표현이 달라도 의미가 비슷한 문장의 유사도가 훨씬 높게 나와야 함");
        System.out.println("(임베딩이 '단어 일치'가 아니라 '의미'를 보고 있다는 증거)");
    }

    static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static String previewFirstN(float[] vector, int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(n, vector.length); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", vector[i]));
        }
        return sb.append(", ...]").toString();
    }
}
