package com.lecture.rag.lab23;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

/**
 * 검색을 "항상 실행되는 Advisor"가 아니라 "LLM이 필요할 때만 호출하는 도구"로 노출한 것.
 * QuestionAnswerAdvisor와 달리, 이 도구는 모델이 스스로 "이 질문엔 문서 검색이 필요하다"고
 * 판단했을 때만 실제로 호출된다 — 일반 잡담 질문에는 호출 안 될 수 있음.
 */
public class DocumentSearchTool {

    private final VectorStore vectorStore;
    private int callCount = 0;

    public DocumentSearchTool(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    public int getCallCount() {
        return callCount;
    }

    @Tool(description = "커피메이커 MCM-200 제품 매뉴얼에서 질문과 관련된 내용을 검색한다. "
            + "제품 사양, 사용법, 환불/AS 정책, 오류 코드 등 매뉴얼에 있을 법한 질문에만 사용할 것.")
    public String searchManual(@ToolParam(description = "매뉴얼에서 검색할 질문 또는 키워드") String query) {
        callCount++;
        System.out.println("  >>> [도구 호출됨] searchManual(\"" + query + "\")");

        var results = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(3).build());

        if (results.isEmpty()) {
            return "매뉴얼에서 관련 내용을 찾지 못했습니다.";
        }

        StringBuilder sb = new StringBuilder();
        for (var doc : results) {
            sb.append("- ").append(doc.getText().replaceAll("\\s+", " ")).append("\n");
        }
        return sb.toString();
    }
}
