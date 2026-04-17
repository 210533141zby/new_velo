from __future__ import annotations

from app.core.config import settings
from app.services.rag.rerank_service import normalize_lookup_text
from app.services.rag.text_utils import compact_text

MODEL_QUERY_KEYWORDS = (
    '什么模型',
    '哪个模型',
    '啥模型',
    '用的模型',
    '你是什么模型',
    '你用什么模型',
    'whatmodel',
    'whichmodel',
)


def is_model_identity_query(query: str) -> bool:
    normalized_query = normalize_lookup_text(query)
    return any(keyword in normalized_query for keyword in MODEL_QUERY_KEYWORDS)


def build_assistant_identity_answer(for_rag: bool) -> str:
    base_answer = f'当前对话使用的模型配置是 {settings.CHAT_MODEL}。'
    if for_rag:
        return f'{base_answer} 这个问题属于系统信息，不依赖知识库检索，因此不附参考引用。'
    return base_answer


def build_no_context_answer() -> str:
    return '根据当前检索到的知识库内容，没有找到足够相关的参考资料，因此我暂时无法给出可靠回答。'


def build_document_judge_prompt(query: str, title: str, content: str) -> str:
    return f"""
    你是 RAG 证据审核器。请先在心里完成主题判断、属性/关系判断和证据判断，再只输出 JSON，不要输出任何解释、标题、代码块或额外文本。

    JSON 结构如下：
    {{
      "core_topic_match": "yes/no",
      "contains_direct_evidence": "yes/no",
      "answerable": "yes/no",
      "evidence_quote": "若能回答，摘录最关键原文；否则留空",
      "answer_brief": "若能回答，用一句简体中文直接回答；否则留空",
      "reason": "不超过18字"
    }}

    判定规则：
    1. core_topic_match 判断文档核心是不是在讨论用户真正关心的对象，而不是只顺带提到某个地名、人名或实体。
    2. contains_direct_evidence 判断文档里是否存在可以直接回答用户问题的证据，而不是只有模糊背景信息。
    3. answerable 只有在前两项都满足时才可以是 yes。
    4. 如果问题是“X的Y”，必须确认 Y 讨论的是 X 本身，而不是别的主体。
    5. 如果问题是关系型问题，必须确认文档真正讨论了两侧实体之间的关系，而不是只提到其中一方。
    6. 如果关系型问题是在问“人物与机构/地点/项目的关系”，只要文档明确写出该人物在该对象中的职责、参与方式、贡献、负责事项、设计内容、任职信息或合作方式，就应判 contains_direct_evidence=yes。
    7. 如果用户问题只是一个人物名、地点名、机构名或其他单独实体名，表示用户在问“这个对象是谁/是什么/做过什么”。这时只要文档主体明确在介绍该实体的身份、角色、经历、贡献、时间信息或关键事实，就应判 core_topic_match=yes 且 contains_direct_evidence=yes。
    8. 对于这种“单实体查询”，不要擅自改写成“该实体与文档标题、机构、地点之间的关系是什么”。只有当文档只是顺带提到该实体、没有围绕该实体给出明确事实时，才判 no。
    9. 如果文档原文已经给出可直接复述的关系句，不要因为用户用了“关系是什么”这种抽象问法就判成 no。
    10. 如果问题是比较两个实体、询问多人的共同点，或者询问文中提到哪些人物及他们与某对象的关系，只要文档中明确提到了其中至少一个相关人物或实体，并给出了身份、职责、参与方式、经历、贡献或关系信息，就应判定 contains_direct_evidence 为 yes。
    11. evidence_quote 必须来自文档原文，不能编造，不超过 80 个字。
    12. answer_brief 只能根据 evidence_quote 组织答案，不能补充文档里没有的事实。

    用户问题：
    {query}

    文档标题：
    {title}

    文档内容：
    {content}
    """


def build_general_rag_prompt(query: str, context: str, warning: str = '') -> str:
    warning_block = f'\n            额外提示：\n            {warning}\n' if warning else ''
    return f"""
            你是一个中文知识库助手，请严格遵守以下规则：
            1. 只能使用简体中文回答。
            2. 优先根据提供的参考上下文回答，不要编造上下文里没有的信息。
            3. 如果参考上下文不足以支持答案，要明确说明“根据当前检索到的知识库内容，我无法确定”。
            4. 不要把无关文档当作依据，不要输出英文回答。
            5. 不要复述不存在于参考上下文中的模型信息或系统设定。
            6. 如果用户问的是某个主题、地点、人物的历史、概况或背景，请先概括与该对象直接相关的主线信息，例如地理位置、主要活动或产业、关键事件、时间线，再补充资料来源和边界。
            7. 如果参考上下文里明确出现了主要产业、活动或领域词，例如“渔业”“航运”“矿石转运”，回答时尽量保留这些原词，不要只用过于泛化的概括替代。
            8. 不要把馆长年龄、机构成立年份、捐赠花絮、通知消息之类的枝节信息放在历史/概况类问题的开头。
            9. 如果上下文主要来自档案馆、论文、新闻或说明文档，也要先回答用户问的对象本身，再说明这些资料来自哪里。
            10. 如果参考上下文只是顺带提到问题中的实体，但没有直接讨论用户询问的主题、属性或关系，必须明确拒答，不能拿顺带提及的句子充当答案。

            参考上下文：
            {context}
            {warning_block}

            用户问题：
            {query}

            回答：
            """


def build_structured_rag_prompt(query: str, context: str) -> str:
    return f"""
            你是一个中文知识库助手，请严格根据参考资料回答用户问题。
            1. 只能使用简体中文。
            2. 只回答用户真正问的事实，不要展开成长篇背景介绍。
            3. 优先输出 1 到 3 句的直接答案。
            4. 不能编造参考资料里没有的事实。
            5. 如果资料不足以直接回答，就明确回答“根据当前检索到的知识库内容，我无法确定。”

            参考上下文：
            {context}

            用户问题：
            {query}

            回答：
            """


def build_exact_document_prompt(query: str, title: str, content: str) -> str:
    return f"""
            你是一个中文知识库助手。用户当前明确在询问文档《{title}》。
            请严格遵守以下规则：
            1. 只能使用简体中文回答。
            2. 只能依据这篇文档的内容回答，不要引入其他文档或系统设定。
            3. 文档标题只用于定位文档，不代表文档事实，不得根据标题自行推断内容。
            4. 如果用户问“讲了什么”“主要内容是什么”“写了什么”，请直接概括这篇文档的核心内容。
            5. 如果用户问的是文档中的具体事实，请直接给出这篇文档能支持的答案，不要绕弯。
            6. 只有在这篇文档确实没有提供所问信息时，才回答“这篇文档没有提供该信息。”
            7. 不要输出英文，不要回答“根据当前检索到的知识库内容，我无法确定”这类泛化拒答。

            文档标题：
            {title}

            文档内容：
            {compact_text(content)}

            用户问题：
            {query}

            回答：
            """
