package cn.bugstack.xfg.dev.tech.test;


import com.alibaba.fastjson.JSON;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@RunWith(SpringRunner.class)
@ActiveProfiles("dev")
@SpringBootTest
public class RAGTest {

    @Resource
    private OllamaChatClient ollamaChatClient;
    @Resource
    private TokenTextSplitter  tokenTextSplitter;  // 文本分割器：把长文本切成小段（AI处理不了太长的文本）
    @Resource
    private PgVectorStore pgVectorStore;

    /**
     * 核心方法 1：upload () —— 把本地文件的知识存到数据库
     */
    @Test
    public void upload(){
        // 1. 读取本地文件：找到./data/file.text这个文件，创建一个“阅读器”
        TikaDocumentReader reader = new TikaDocumentReader("./data/file.text");
        // 2. 把文件内容读出来，变成“文档对象”（Document）：就像把文件里的字复制到笔记本上
        List<Document> documents = reader.get();
        // 3. 分割文本：AI处理文本有长度限制，把长文本切成小段（比如把1000字切成5段200字）
        List<Document> documentList = tokenTextSplitter.apply(documents);
        // 4. 给文档贴“标签”：给所有文档加个标记“knowledge=知识库名称”
        // 目的：后面查的时候，只找这个标签下的内容，避免混了其他知识
        documents.forEach(document -> document.getMetadata().put("knowledge", "人物"));
        documentList.forEach(document -> document.getMetadata().put("knowledge", "人物"));
        // 5. 把切好的小段文档存到PostgreSQL数据库里
        // 数据库会把文本转换成“数字串”（向量），方便后面快速找相关内容
        pgVectorStore.add(documentList);


        // 6. 打印日志：告诉我们“知识已经存好了”
        log.info("上传完成");
    }


    @Test
    public void chat(){
        String question = "介绍一下陈弘棣？";

        // 2. 给AI定“规矩”（系统提示词）：
        // 大白话翻译：必须用DOCUMENTS里的信息回答，要准确；不知道就说不知道；回答必须用中文
        String SYSTEM_PROMPT = """
        你是一个基于知识库回答问题的助手。
        你必须严格根据 DOCUMENTS 中提供的内容回答问题。
        如果 DOCUMENTS 中没有明确答案，请直接回答：根据已知信息无法回答。
        禁止使用你自己的知识进行补充、猜测或编造。
        回答必须使用中文。
        
        DOCUMENTS:
        {documents}
    """;


        // 3. 创建“搜索请求”：去数据库里找和问题相关的内容
        // withTopK(5)：只找最相关的5条；withFilterExpression：只找标签是“知识库名称”的内容
        SearchRequest request = SearchRequest.query(question).withTopK(5).withFilterExpression("knowledge == '人物'");

        // 4. 执行搜索：从数据库里找出和“朱昊，今年多少岁？”最相关的5条内容
        List<Document> documents = pgVectorStore.similaritySearch(request);

        // 5. 把搜到的5条内容拼接成一个长字符串：方便传给AI看
        String documentsCollectors = documents.stream()
                .map(Document::getContent)
                .collect(Collectors.joining("\n"));

        // 6. 把“规矩”里的占位符替换成搜到的内容：生成给AI的“参考资料”
        Message ragmessage = new SystemPromptTemplate(SYSTEM_PROMPT).createMessage(Map.of("documents", documentsCollectors));

        // 7. 把“用户问题”和“参考资料”打包：准备发给AI
        ArrayList<Message> messages = new ArrayList<>();
        messages.add(ragmessage);
        messages.add(new UserMessage(question));

        // 8. 调用AI模型：把打包好的内容发给本地的deepseek-r1:1.5b模型，要回答
        ChatResponse chatResponse = ollamaChatClient.call(new Prompt(messages, OllamaOptions.create().withModel("deepseek-r1:1.5b")));

        // 9. 打印AI的回答：方便看结果
// ===== 用户提问 =====
        log.info("\n================ 对话开始 ================");

// 打印用户问题
        log.info("【用户】");
        log.info("{}", question);
        log.info("");

//// ===== 检索结果 =====
//        log.info("【知识库检索】");
//        log.info("命中 {} 条相关内容", documents.size());
//
//// 遍历打印检索内容
//        int index = 1;
//        for (Object doc : documents) {
//            String content = doc.toString();
//            log.info("证据 {}: {}", index++, content);
//        }
//
//        log.info("");

// ===== AI回答 =====
        String answer = chatResponse.getResult().getOutput().getContent();

        log.info("【AI】");
        log.info("{}", answer);

        log.info("================ 对话结束 ================\n");
    }



    @Autowired
    private org.springframework.core.env.Environment environment;

    @Autowired
    private javax.sql.DataSource dataSource;

    @Test
    public void printDbConfig() throws Exception {
        System.out.println("activeProfiles=" + java.util.Arrays.toString(environment.getActiveProfiles()));
        System.out.println("url=" + environment.getProperty("spring.datasource.url"));
        System.out.println("username=" + environment.getProperty("spring.datasource.username"));
        System.out.println("password=" + environment.getProperty("spring.datasource.password"));
        System.out.println("dataSource=" + dataSource);

        try (java.sql.Connection conn = dataSource.getConnection()) {
            System.out.println("db url=" + conn.getMetaData().getURL());
            System.out.println("db user=" + conn.getMetaData().getUserName());
            System.out.println("ok");
        }
    }

}
