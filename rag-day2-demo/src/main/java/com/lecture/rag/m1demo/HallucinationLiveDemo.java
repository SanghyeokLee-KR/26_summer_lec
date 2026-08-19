package com.lecture.rag.m1demo;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * M1.1 항목 2~4 라이브 재현용 — LLM의 한계(Hallucination, Knowledge cutoff) 시연
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=m1-hallucination
 *
 * 강의 포인트: 모델이 "모릅니다"라고 하지 않고 그럴듯한 답을 지어내는 순간을 학생들에게 직접 보여줄 것.
 * 질문은 사용 중인 로컬 모델의 실제 지식 범위에 따라 강사가 강의 전 미리 테스트해서 잘 걸리는 질문으로 교체할 것.
 */
@Component
@Profile("m1-hallucination")
public class HallucinationLiveDemo implements CommandLineRunner {

    private final ChatModel chatModel;

    public HallucinationLiveDemo(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    @Override
    public void run(String... args) {
        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 한국어로 답변하세요.")
                .build();

        String[] demoQuestions = {
                // 존재하지 않는 API를 지어내는지 확인
                "Spring AI의 VectorStore 인터페이스에 있는 deleteAllByMetadata() 메서드 사용법을 알려줘.",
                // 최신 이벤트 / knowledge cutoff 확인
                "어제 발표된 최신 Spring Boot 패치 버전이 뭐야?",
                // 사내/프라이빗 데이터 — 모델이 알 수 없는 정보
                "우리 학교 2026학년도 2학기 수강신청 정정기간이 며칠이야?"
        };

        for (String question : demoQuestions) {
            System.out.println("질문: " + question);
            String answer = chatClient.prompt().user(question).call().content();
            System.out.println("답변: " + answer);
            System.out.println("----");
        }

        System.out.println("관찰 포인트: 모델이 '모른다'고 하지 않고 그럴듯한 이름/날짜/절차를 지어내는지 확인.");
    }
}
