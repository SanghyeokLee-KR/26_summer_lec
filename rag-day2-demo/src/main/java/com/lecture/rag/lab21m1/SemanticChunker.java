package com.lecture.rag.lab21m1;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;

import java.util.ArrayList;
import java.util.List;

/**
 * M2.1 — Semantic Chunking. 문장 단위로 쪼갠 뒤 각 문장을 임베딩하고, 인접 문장 간 코사인 유사도가
 * 급락하는 지점을 청크 경계로 채택한다. 문장마다 임베딩 호출이 필요해 비용·시간이 더 든다는 트레이드오프가 있음.
 */
public class SemanticChunker {

    private final EmbeddingModel embeddingModel;
    private final double dropThreshold;

    public SemanticChunker(EmbeddingModel embeddingModel, double dropThreshold) {
        this.embeddingModel = embeddingModel;
        this.dropThreshold = dropThreshold;
    }

    public record SentenceSimilarity(String sentence, double similarityToPrev) {}

    public List<SentenceSimilarity> analyze(Document doc) {
        String[] sentences = doc.getText().split("(?<=[.!?다]\\s)|\n+");
        List<String> clean = new ArrayList<>();
        for (String s : sentences) {
            if (!s.isBlank()) clean.add(s.trim());
        }

        List<SentenceSimilarity> result = new ArrayList<>();
        float[] prevVec = null;
        for (String sentence : clean) {
            float[] vec = embeddingModel.embed(sentence);
            double sim = (prevVec == null) ? 1.0 : cosine(prevVec, vec);
            result.add(new SentenceSimilarity(sentence, sim));
            prevVec = vec;
        }
        return result;
    }

    public List<Document> split(Document doc) {
        List<SentenceSimilarity> analyzed = analyze(doc);
        List<Document> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (SentenceSimilarity s : analyzed) {
            if (s.similarityToPrev() < dropThreshold && !current.isEmpty()) {
                chunks.add(new Document(current.toString().trim()));
                current = new StringBuilder();
            }
            current.append(s.sentence()).append(" ");
        }
        if (!current.isEmpty()) chunks.add(new Document(current.toString().trim()));
        return chunks;
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
