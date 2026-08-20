package com.lecture.rag.dbquery;

import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/**
 * vector_store 테이블을 콘솔에서 바로 들여다보는 조회 도구. 여러 데모/학생 실습이 같은 테이블을
 * 공유하다 보니 metadata 스키마가 제각각인 경우가 많아(source, file_name 등), DBeaver 켤 것 없이
 * "지금 뭐가 들어있는지"부터 빠르게 확인하려고 만듦.
 * 실행: ./mvnw spring-boot:run -Dspring-boot.run.profiles=db-query
 */
@Component
@Profile("db-query")
public class DbQueryTool implements CommandLineRunner {

    private final JdbcTemplate jdbcTemplate;

    public DbQueryTool(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public void run(String... args) {
        Long total = jdbcTemplate.queryForObject("SELECT count(*) FROM vector_store", Long.class);
        System.out.println("=== vector_store 총 문서 수: " + total + " ===");

        System.out.println("--- file_name별 문서 수 ---");
        printCounts("file_name");

        System.out.println("--- source별 문서 수 (커스텀 인덱서가 넣은 값, 없으면 전부 null) ---");
        printCounts("source");
        System.out.println();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("검색어(내용에 포함된 문자열, 빈 줄 입력 시 종료)> ");
            String keyword = scanner.nextLine();
            if (keyword == null || keyword.isBlank()) break;

            List<Map<String, Object>> rows = jdbcTemplate.queryForList(
                    "SELECT id, metadata->>'file_name' AS file_name, metadata->>'source' AS source, "
                            + "left(content, 80) AS preview FROM vector_store WHERE content ILIKE ? LIMIT 10",
                    "%" + keyword + "%");

            if (rows.isEmpty()) {
                System.out.println("  결과 없음");
            }
            for (Map<String, Object> row : rows) {
                System.out.printf("  [%s] file_name=%s source=%s | %s...%n",
                        row.get("id"), row.get("file_name"), row.get("source"), row.get("preview"));
            }
            System.out.println();
        }
    }

    private void printCounts(String metadataKey) {
        List<Map<String, Object>> rows = jdbcTemplate.queryForList(
                "SELECT metadata->>'" + metadataKey + "' AS k, count(*) AS cnt "
                        + "FROM vector_store GROUP BY metadata->>'" + metadataKey + "' ORDER BY cnt DESC");
        for (Map<String, Object> row : rows) {
            System.out.printf("  %-30s %s건%n", row.get("k"), row.get("cnt"));
        }
    }
}
