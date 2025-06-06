import jieba
from hanlp_restful import HanLPClient
from jieba import posseg

texts = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"


'''
    分词：将连续的 字序列 按照一定的规范重新组合成 词序列 的过程。
    在英文的行文中，单词之间是以空格作为自然分界符的，
        而中文只是字、句和段能通过明显的分界符来简单划界，唯独词没有一个形式上的分界符。
        分词过程就是找到这样分界符的过程。
        
    分词的作用：
        预处理：分词是文本处理的第一步，能够将文本分解成有意义的单元，为后续的分析提供基础。
        理解结构：分词有助于理解句子的基本构成和含义，尤其是在做文本分类、情感分析等任务时，分词是不可缺少的一步。

    常用的中文分词工具包括 Jieba、HanLP 等。
'''
'''
    Jieba分词工具
        Jieba（"结巴"）是一个开源的Python中文分词组件，支持精确模式、全模式和搜索引擎模式三种分词模式。
    
    Jieba的主要特点：
        支持多种分词模式： 精确模式、全模式和搜索引擎模式，满足不同场景的需求。
        支持自定义词典： 用户可以添加自定义的词语，提高分词准确率。
        支持词性标注： 可以为每个词语标注词性，例如名词、动词等。
        支持关键词提取： 可以提取文本中的关键词。
        支持并行分词： 可以利用多核处理器加速分词。
        简单易用： API 简单明了，易于上手。
        开源免费： 任何人都可以免费使用。
        
        
'''
def demo_cutword():

    # 精确模式：试图将句子最精确地切开，适合文本分析。 - 默认是精确模式
    #   sentence(句子) 参数为：要被切分的句子
    print(list(jieba.cut(sentence=texts)))
    print(jieba.lcut(sentence=texts))     # lcut() 方法，本质上就是 cut()方法,后经过list()转化

    # 全模式：将句子中所有可以成词的词语都扫描出来，速度非常快，但是不能消除歧义。
    # print(jieba.lcut(sentence=texts, cut_all=True))

    # 搜索引擎模式：在精确模式的基础上，对长词再次切分，进行细粒度分词，提高召回率,适合用于搜索引擎分词。
    # print(list(jieba.cut_for_search(sentence=texts)))
    # print(jieba.lcut_for_search(sentence=texts))

    # 繁体分词： 针对中国香港, 台湾地区的繁体文本进行分词。
    # content = "煩惱即是菩提，我暫且不提"
    # print(jieba.lcut(sentence=content))

    # 使用用户自定义的词典：
    #   添加自定义词典后, jieba能够准确识别词典中出现的词汇，提升整体的识别准确率。
    #   jieba分词器默认的词典是：jieba/dict.txt
    # jieba.load_userdict("dict.txt")
    # print(jieba.lcut(sentence=texts))


'''
    命名实体识别：
        命名实体识别（NER）是自然语言处理中的一个任务，
            旨在从文本中识别出特定类别的实体（如人名、地名、机构名、日期、时间等）。
        NER是信息抽取的一部分，帮助计算机识别出与任务相关的实体信息。
    
    命名实体识别作用：
        信息抽取：NER帮助从海量的文本中自动抽取出结构化的实体信息，为数据分析、问答系统等提供有价值的内容。
        问答系统：在智能问答系统中，NER能够帮助系统准确理解用户的提问，并提取相关的实体信息以便生成更准确的回答。
        文本理解：NER对于文本理解至关重要，它帮助系统识别出文本中的关键信息，
            例如人物、地点、组织等，进而为语义分析和事件抽取提供支持。
    
    处理工具：
        SpaCy、NLTK、Stanford NER、BERT（通过微调）、LTP、**HanLP**等都可以用于命名实体识别任务。
        
'''
def demo_ner():



    # HanLP：用于命名实体识别任务
    hanlp = HanLPClient(url="https://www.hanlp.com/api", auth=None, language="zh")
    print(hanlp.parse('鲁迅, 浙江绍兴人, 五四新文化运动的重要参与者, 代表作朝花夕拾.', tasks=['ner']))


'''
    词性标注(Part-Of-Speech tagging, 简称POS)就是为文本中的每个词分配一个语法类别（即词性），
            例如名词、动词、形容词等。
    词性标注能够帮助模型理解词汇在句子中的语法功能，并为进一步的句法分析和语义分析提供支持。
    
    作用：
        1.理解句子结构：通过词性标注，可以知道每个词在句子中的角色，帮助理解句子的语法结构。
        2.支持其他NLP任务：许多高级任务如命名实体识别（NER）、句法分析、情感分析等，通常依赖于词性标注的结果。
        3.歧义消解：词性标注有助于解决同一单词在不同上下文中可能具有不同词性的情况。
                例如，单词 "lead" 可能是动词（引导）也可能是名词（铅），通过词性标注可以解决这种歧义。
                
    处理工具：
        Jieba、NLTK、SpaCy、**Stanford POS Tagger**等是常用的词性标注工具。'''
def demo_posseg():

    # 步骤 1: 分词并词性标注
    # 结果返回一个装有pair元组的列表, 每个pair元组中分别是词汇及其对应的词性,
    #   具体词性含义请参照[附录: jieba词性对照表]()
    words = posseg.lcut(sentence=texts)

    res = []
    # 提取命名实体（人名、地名、组织机构名）
    for word, flag in words:
        if flag in ['n', 'v', 'ns']:        # n：名词, r: 代词, v:动词, ns: 地名
            res.append((word, flag))

    print(res)


if __name__ == '__main__':
    demo_cutword()
    # demo_ner()
    # demo_posseg()