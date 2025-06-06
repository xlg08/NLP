'''

'''

# 需要科学上网才能访问谷歌翻译
from googletrans import Translator, LANGUAGES
# pip install googletrans==4.0.0-rc1 -i https://mirrors.aliyun.com/pypi/simple/

def back_translate(text, src_lang='zh-cn', intermediate_lang='en'):
    # 回译数据增强
    # param text: 原始文本
    # param src_lang: 原始语言（默认中文）
    # param intermediate_lang: 中间语言（默认英语）
    # return: 回译后的文本
    translator = Translator()

    # 将原始文本翻译为中间语言
    translated = translator.translate(text=text, src=src_lang, dest=intermediate_lang)
    intermediate_text = translated.text

    # 将中间语言翻译回原始语言
    back_translated = translator.translate(intermediate_text, src=intermediate_lang, dest=src_lang)
    back_translated_text = back_translated.text

    return back_translated_text


if __name__ == '__main__':

    # 示例
    # original_text = '5月19日到20日，习近平总书记赴河南考察调研。从智能工厂生产线到文化遗产保护地，从夯实制造业基础到文化传承保护.'

    # original_text = "这个价格非常便宜!"
    # augmented_text = back_translate(original_text)
    # print("原始文本：", original_text)
    # print("回译文本：", augmented_text)

    print(LANGUAGES)


