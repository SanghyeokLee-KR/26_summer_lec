package com.lecture.rag.m1demo;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * M1.1 항목 10 — 미니 라이브 데모: 같은 질문을 (a) 순수 LLM (b) RAG 챗봇에 던져서 답변 차이를 비교
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=m1-compare
 */
@Component
@Profile("m1-compare")
public class PureLlmVsRagDemo implements CommandLineRunner {

    private final ChatModel chatModel;
    private final EmbeddingModel embeddingModel;
    private final String documentPath;

    public PureLlmVsRagDemo(ChatModel chatModel,
                             EmbeddingModel embeddingModel,
                             @Value("${rag.demo.document-path}") String documentPath) {
        this.chatModel = chatModel;
        this.embeddingModel = embeddingModel;
        this.documentPath = documentPath;
    }

    @Override
    public void run(String... args) {
        String question = "MCM-200 커피메이커의 E2 오류 코드가 뜨면 어떻게 해야 해?";

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 한국어로 답변하세요.")
                .build();

        // (a) 순수 LLM — 문서에 없는 제품이므로 모델이 지어내거나 얼버무릴 가능성이 높음
        String pureAnswer = chatClient.prompt().user(question).call().content();

        // (b) RAG 챗봇 — 실습 문서(manual.txt/pdf)를 검색해서 근거 있는 답변 생성
        VectorStore vectorStore = buildVectorStore();
        QuestionAnswerAdvisor qaAdvisor = QuestionAnswerAdvisor.builder(vectorStore).build();
        String ragAnswer = chatClient.prompt()
                .advisors(qaAdvisor)
                .user(question)
                .call()
                .content();

        System.out.println("질문: " + question);
        System.out.println();
        System.out.println("[a] 순수 LLM 답변:");
        System.out.println(pureAnswer);
        System.out.println();
        System.out.println("[b] RAG 챗봇 답변 (실습 문서 근거):");
        System.out.println(ragAnswer);
        System.out.println();
        System.out.println("강의 포인트: (b)는 문서 제5조의 실제 오류 코드 설명(고객센터 문의)에 근거해 답하고,");
        System.out.println("(a)는 실제 존재하지 않는 제품이므로 지어내거나 일반론으로 얼버무리는 차이를 짚어줄 것.");
    }

    private VectorStore buildVectorStore() {
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(documentPath);
        List<Document> documents = pdfReader.get();

        TokenTextSplitter splitter = TokenTextSplitter.builder()
                .withChunkSize(300)
                .build();
        List<Document> chunks = splitter.apply(documents);

        VectorStore vectorStore = SimpleVectorStore.builder(embeddingModel).build();
        vectorStore.add(chunks);
        return vectorStore;
    }
}
