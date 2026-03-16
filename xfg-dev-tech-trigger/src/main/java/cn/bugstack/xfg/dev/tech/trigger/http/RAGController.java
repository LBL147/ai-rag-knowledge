package cn.bugstack.xfg.dev.tech.trigger.http;

import cn.bugstack.xfg.dev.tech.api.IRAGService;
import cn.bugstack.xfg.dev.tech.api.response.Response;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RList;
import org.redisson.api.RedissonClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/v1/rag/")
@CrossOrigin("*")
@Slf4j
public class RAGController implements IRAGService {


    @Resource
    private OllamaChatClient ollamaChatClient;
    @Resource
    private TokenTextSplitter tokenTextSplitter;
    @Resource
    private SimpleVectorStore simpleVectorStore;
    @Resource
    private PgVectorStore pgVectorStore;
    @Resource
    private RedissonClient redissonClient;


    @RequestMapping(value = "query_rag_tag_list", method = RequestMethod.GET)
    @Override
    public Response<List<String>> queryRagTagList() {
        RList<String> elements=redissonClient.getList("ragTag");  //从 Redis 里面取一个 List

        return Response.<List<String>>builder()
                .code("0000")
                .info("调用成功")
                .data(elements)
                .build();
    }

    @RequestMapping(value = "file/upload", method = RequestMethod.POST)
    @Override
    public Response<String> uploadFile(@RequestParam String ragTag, @RequestParam("file") List<MultipartFile> files) {
        log.info("上传知识库开始 {}", ragTag);
        for(MultipartFile file : files){
            // 使用 Tika 解析文件
            // 可以读取 pdf、doc、txt 等文件
            TikaDocumentReader documentReader = new TikaDocumentReader(file.getResource());
            // 获取解析后的文本内容
            // 每个 Document 是一段文本
            List<Document> documents = documentReader.get();
            // 获取解析后的文本内容
            // 每个 Document 是一段文本
            List<Document> documentList = tokenTextSplitter.apply(documents);

            // 给原始文本添加 metadata
            // metadata 里记录知识库标签
            for (Document document : documents) {
                document.getMetadata().put("knowledge", ragTag);
            }
            // 给切分后的文本也添加标签
            documentList.forEach(doc ->
                    doc.getMetadata().put("knowledge", ragTag));

            pgVectorStore.accept(documentList);
            // 把切分后的文本存入向量数据库
            //
            // 过程其实是：
            // 文本 -> embedding 向量 -> 存入 pgvector


            // 从 Redis 获取知识库标签列表
            RList<String> elements = redissonClient.getList("ragTag");

            if (!elements.contains(ragTag)) {
                // 添加知识库标签
                elements.add(ragTag);
            }
            //Redis 不是用来存文件内容的，也 不是用来存向量数据的。
            //用来记录“当前系统里都有哪些知识库标签”

        }
        log.info("上传知识库结束 {}", ragTag);
        return Response.<String>builder()
                .code("0000")
                .info("调用成功")
                .build();
        // 返回接口结果
    }
}
