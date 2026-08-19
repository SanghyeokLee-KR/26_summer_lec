package com.lecture.rag.api;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PostConstruct;
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
import org.springframework.context.annotation.Profile;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

/**
 * Swagger UI로 라이브 데모를 보여주기 위한 REST 엔드포인트.
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=api
 * 접속: http://localhost:8080/swagger-ui.html
 */
@RestController
@Profile("api")
@RequestMapping("/api")
@Tag(name = "RAG 라이브 데모", description = "M1.1 강의용 라이브 데모 — 순수 LLM vs RAG 비교, 할루시네이션 재현, RAG 챗봇")
public class LiveDemoController {

    private final ChatModel chatModel;
    private final EmbeddingModel embeddingModel;
    private final String documentPath;
    private ChatClient chatClient;
    private VectorStore vectorStore;

    public LiveDemoController(ChatModel chatModel,
                               EmbeddingModel embeddingModel,
                               @Value("${rag.demo.document-path}") String documentPath) {
        this.chatModel = chatModel;
        this.embeddingModel = embeddingModel;
        this.documentPath = documentPath;
    }

    @PostConstruct
    void init() {
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 한국어로 답변하세요.")
                .build();

        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(documentPath);
        List<Document> documents = pdfReader.get();
        TokenTextSplitter splitter = TokenTextSplitter.builder().withChunkSize(300).build();
        List<Document> chunks = splitter.apply(documents);

        this.vectorStore = SimpleVectorStore.builder(embeddingModel).build();
        this.vectorStore.add(chunks);
    }

    public record CompareResponse(String question, String pureAnswer, String ragAnswer) {}

    public record QaResult(String question, String answer) {}

    @Operation(summary = "순수 LLM vs RAG 챗봇 비교",
            description = "같은 질문을 (a) 문서 없이 순수 LLM에게, (b) 실습 문서를 검색하는 RAG 챗봇에게 던져서 답변을 나란히 비교한다. "
                    + "질문은 우리가 실습 중인 가상 제품 'MCM-200 커피메이커 매뉴얼'(제1조~제7조, 오류코드 E1~E3, 고객센터 1544-0000)에 있는 내용으로 물어봐야 대비가 잘 보인다.")
    @GetMapping("/compare")
    public CompareResponse compare(
            @Parameter(description = "실습 문서(MCM-200 매뉴얼) 관련 질문. 예: 오류 코드, 보증 기간, 세척 방법 등 문서에 실제로 답이 있는 질문일수록 순수 LLM과의 차이가 극적으로 드러남.",
                    example = "MCM-200 커피메이커의 E2 오류 코드가 뜨면 어떻게 해야 해?")
            @RequestParam(defaultValue = "MCM-200 커피메이커의 E2 오류 코드가 뜨면 어떻게 해야 해?") String question) {

        String pureAnswer = chatClient.prompt().user(question).call().content();

        QuestionAnswerAdvisor qaAdvisor = QuestionAnswerAdvisor.builder(vectorStore).build();
        String ragAnswer = chatClient.prompt()
                .advisors(qaAdvisor)
                .user(question)
                .call()
                .content();

        return new CompareResponse(question, pureAnswer, ragAnswer);
    }

    @Operation(summary = "할루시네이션 재현 (프리셋 질문 3개 + 직접 입력 1개)",
            description = "존재하지 않는 API, 최신 이벤트, 사내(학교) 데이터에 대한 질문을 순수 LLM에게 던져 모델이 어떻게 지어내는지 보여준다. "
                    + "학생이 즉석에서 제안하는 '이것도 지어낼까?' 질문을 studentQuestion에 넣으면 4번째 결과로 같이 나온다 (모델이 절대 알 수 없는, 방금 지어낸 듯한 질문일수록 효과적).")
    @GetMapping("/hallucination")
    public List<QaResult> hallucination(
            @Parameter(description = "(선택) 학생이 직접 만들어보는 할루시네이션 유도 질문. 비워두면 프리셋 3개만 실행됨. "
                    + "예: 존재하지 않는 지어낸 라이브러리/인물/사건에 대해 물어보면 잘 걸림.",
                    example = "우리 학교 총학생회장 이름이 뭐야?")
            @RequestParam(required = false) String studentQuestion) {
        List<String> questions = new ArrayList<>(List.of(
                "Spring AI의 VectorStore 인터페이스에 있는 deleteAllByMetadata() 메서드 사용법을 알려줘.",
                "어제 발표된 최신 Spring Boot 패치 버전이 뭐야?",
                "우리 학교 2026학년도 2학기 수강신청 정정기간이 며칠이야?"
        ));
        if (studentQuestion != null && !studentQuestion.isBlank()) {
            questions.add(studentQuestion);
        }
        return questions.stream()
                .map(q -> new QaResult(q, chatClient.prompt().user(q).call().content()))
                .toList();
    }

    @Operation(summary = "RAG 챗봇에게 질문하기",
            description = "실습 문서(MCM-200 커피메이커 매뉴얼)를 검색해서 근거 있는 답변을 생성한다. 자유롭게 질문을 바꿔서 테스트해볼 것.")
    @GetMapping("/chat")
    public QaResult chat(
            @Parameter(description = "MCM-200 매뉴얼에 있는 내용으로 자유롭게 질문. 문서에 없는 내용을 사실인 것처럼 물어보면 '모른다고 답하기'를 아직 안 배운 상태라 여전히 지어낼 수 있음 (Day2/M2.6에서 해결).",
                    example = "물탱크 용량이 얼마야?")
            @RequestParam(defaultValue = "물탱크 용량이 얼마야?") String question) {
        QuestionAnswerAdvisor qaAdvisor = QuestionAnswerAdvisor.builder(vectorStore).build();
        String answer = chatClient.prompt()
                .advisors(qaAdvisor)
                .user(question)
                .call()
                .content();
        return new QaResult(question, answer);
    }
}
