package com.lecture.rag.lab23;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Lab 2.3 — (A) 프롬프트 바인딩형 RAG(QuestionAnswerAdvisor) vs
 *           (B) 툴로서의 RAG(Agentic RAG, @Tool + DocumentSearchTool) 비교.
 *
 * 같은 두 질문(일반 질문 / 문서 관련 질문)을 두 방식에 각각 던져서,
 * (A)는 항상 검색하고 (B)는 모델이 필요하다고 판단할 때만 검색 도구를 호출하는지 확인한다.
 *
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab23
 * (사전에 lab21을 한 번 실행해서 PGVector에 manual.pdf가 인덱싱되어 있어야 함)
 */
@Component
@Profile("lab23")
public class RagAsToolDemo implements CommandLineRunner {

    private final ChatModel chatModel;
    private final VectorStore vectorStore;

    public RagAsToolDemo(ChatModel chatModel, VectorStore vectorStore) {
        this.chatModel = chatModel;
        this.vectorStore = vectorStore;
    }

    @Override
    public void run(String... args) {
        String generalQuestion = "대한민국의 수도는 어디야?";
        String docQuestion = "환불은 며칠 안에 가능해?";

        System.out.println("################ (A) 프롬프트 바인딩형 RAG (QuestionAnswerAdvisor) ################");
        runPromptBinding(generalQuestion);
        runPromptBinding(docQuestion);

        System.out.println();
        System.out.println("################ (B) 툴로서의 RAG (Agentic RAG, @Tool) ################");
        DocumentSearchTool tool = new DocumentSearchTool(vectorStore);
        runAsTool(tool, generalQuestion);
        runAsTool(tool, docQuestion);
    }

    private void runPromptBinding(String question) {
        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 한국어로 답변하세요.")
                .build();
        QuestionAnswerAdvisor qaAdvisor = QuestionAnswerAdvisor.builder(vectorStore).build();

        System.out.println("[질문] " + question);
        System.out.println("  (QuestionAnswerAdvisor가 붙어있으므로 이 질문과 무관해도 무조건 검색부터 실행됨)");
        String answer = chatClient.prompt()
                .advisors(qaAdvisor)
                .user(question)
                .call()
                .content();
        System.out.println("[답변] " + answer);
        System.out.println();
    }

    private void runAsTool(DocumentSearchTool tool, String question) {
        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 한국어로 답변하세요. 필요할 때만 도구를 사용하세요.")
                .build();

        int before = tool.getCallCount();
        System.out.println("[질문] " + question);
        String answer = chatClient.prompt()
                .tools(tool)
                .user(question)
                .call()
                .content();
        int after = tool.getCallCount();

        System.out.println("[도구 호출 여부] " + (after > before ? "호출됨 (" + (after - before) + "회)" : "호출 안 됨"));
        System.out.println("[답변] " + answer);
        System.out.println();
    }
}
