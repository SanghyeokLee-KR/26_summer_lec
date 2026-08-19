package com.lecture.rag.lab14;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.PathResource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Lab 1.4 — PDF 업로드 → 인덱싱 → RAG 질의응답 API 서버
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=lab14
 * 접속: http://localhost:8080/swagger-ui.html
 *
 * Lab1.2(인덱싱 파이프라인)와 Lab1.3(RAG 챗봇)에서 콘솔로 하던 것을
 * 실제 REST API로 옮긴 것 — 어떤 PDF를 올려도 그 문서 기반으로 답변하는 서버.
 * 준비된 PDF가 없으면 아무 문서(빈 문서, 텍스트 한 장짜리 PDF 등)를 올려서
 * 파이프라인 자체가 동작하는지만 확인해도 됨.
 */
@RestController
@Profile("lab14")
@RequestMapping("/api/lab14")
@Tag(name = "Lab1.4 PDF 업로드 RAG", description = "PDF를 업로드하면 그 문서를 인덱싱하고, 그 내용을 기반으로 질문에 답하는 API")
public class PdfRagApiController {

    private final EmbeddingModel embeddingModel;
    private final ChatModel chatModel;

    private volatile VectorStore vectorStore;
    private volatile String loadedFileName;

    public PdfRagApiController(EmbeddingModel embeddingModel, ChatModel chatModel) {
        this.embeddingModel = embeddingModel;
        this.chatModel = chatModel;
    }

    public record IndexResult(String fileName, int pageCount, int chunkSize, int chunkCount, String message) {}

    public record ChatResult(String fileName, String question, String answer) {}

    @Operation(summary = "PDF 업로드 + 인덱싱 (Load → Split → Embed → Store)",
            description = "PDF 파일을 업로드하면 그 자리에서 Lab1.2와 동일한 인덱싱 파이프라인(문서 로드 → 청킹 → 임베딩 → VectorStore 저장)을 실행한다. "
                    + "새 파일을 업로드하면 이전에 올렸던 문서는 사라지고 새 문서로 완전히 교체된다(한 번에 문서 1개만 유지). "
                    + "테스트할 PDF가 없으면 아무 PDF(빈 문서, 한 페이지짜리 텍스트를 PDF로 저장한 것 등)나 올려서 파이프라인이 도는지만 확인해도 된다.")
    @PostMapping(value = "/index", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public IndexResult index(
            @RequestParam("file") MultipartFile file,
            @Parameter(description = "청킹 시 청크 하나의 최대 토큰 수")
            @RequestParam(defaultValue = "300") int chunkSize) throws IOException {
        Path tempFile = Files.createTempFile("lab14-upload-", ".pdf");
        try {
            file.transferTo(tempFile);

            PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(new PathResource(tempFile));
            List<Document> documents = pdfReader.get();

            TokenTextSplitter splitter = TokenTextSplitter.builder()
                    .withChunkSize(chunkSize)
                    .build();
            List<Document> chunks = splitter.apply(documents);

            VectorStore newVectorStore = SimpleVectorStore.builder(embeddingModel).build();
            newVectorStore.add(chunks);

            this.vectorStore = newVectorStore;
            this.loadedFileName = file.getOriginalFilename();

            return new IndexResult(
                    file.getOriginalFilename(),
                    documents.size(),
                    chunkSize,
                    chunks.size(),
                    "인덱싱 완료 — 이제 /api/lab14/chat 으로 이 문서에 대해 질문할 수 있습니다.");
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @Operation(summary = "업로드한 PDF 기반 질문",
            description = "직전에 /index로 올린 PDF에서 관련 내용을 검색해서 답변한다. "
                    + "아직 업로드한 적이 없으면 안내 메시지를 반환한다(에러 대신 안내).")
    @GetMapping("/chat")
    public ChatResult chat(
            @Parameter(description = "방금 업로드한 문서에 대한 질문. 문서에 실제로 있는 내용을 물어봐야 RAG 효과가 확인됨.",
                    example = "이 문서의 핵심 내용을 세 줄로 요약해줘")
            @RequestParam(defaultValue = "이 문서의 핵심 내용을 세 줄로 요약해줘") String question) {

        VectorStore currentStore = this.vectorStore;
        if (currentStore == null) {
            return new ChatResult(null, question, "아직 업로드된 문서가 없습니다. 먼저 /api/lab14/index 로 PDF를 올려주세요.");
        }

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("항상 한국어로 답변하세요.")
                .build();
        QuestionAnswerAdvisor qaAdvisor = QuestionAnswerAdvisor.builder(currentStore).build();

        String answer = chatClient.prompt()
                .advisors(qaAdvisor)
                .user(question)
                .call()
                .content();

        return new ChatResult(loadedFileName, question, answer);
    }
}
