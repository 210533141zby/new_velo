"""Run a small live RAG regression probe suite against the local backend."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE_URL = 'http://127.0.0.1:8000/api/v1/agent/chat'
NO_CONTEXT_ANSWER = '根据当前检索到的知识库内容，没有找到足够相关的参考资料，因此我暂时无法给出可靠回答。'

CASES = [
    {
        'query': 'RAG测试文档的默认聊天模型是什么',
        'must_include': ['qwen2.5:7b-instruct'],
        'source_titles': ['RAG测试文档'],
    },
    {
        'query': '为什么研究者重视编号为A-17-204的蓝布账册',
        'must_include': ['沈见川'],
        'source_titles': ['雾潮镇的档案馆'],
    },
    {
        'query': '雾潮镇的历史',
        'must_include': ['雾潮镇', '渔业'],
        'must_exclude': ['馆长林若岑今年四十二岁'],
        'source_titles': ['雾潮镇的档案馆'],
    },
    {
        'query': 'Shor算法能解决什么问题',
        'must_include': ['大整数质因数分解'],
        'source_titles': ['量子计算与Shor算法'],
    },
    {
        'query': '请问一下，Shor主要解决什么问题吗？',
        'must_include': ['大整数质因数分解'],
        'source_titles': ['量子计算与Shor算法'],
    },
    {
        'query': '青苗法是什么',
        'must_include': ['青黄不接', '农民贷款'],
        'source_titles': ['北宋熙宁变法中的经济政策'],
    },
    {
        'query': '沈阳',
        'exact_response': NO_CONTEXT_ANSWER,
        'source_titles': [],
    },
    {
        'query': '韩国与新加坡的关系',
        'exact_response': NO_CONTEXT_ANSWER,
        'source_titles': [],
    },
    {
        'query': '韩国',
        'exact_response': NO_CONTEXT_ANSWER,
        'source_titles': [],
    },
    {
        'query': '新加坡的档案馆',
        'exact_response': NO_CONTEXT_ANSWER,
        'source_titles': [],
    },
    {
        'query': '新加坡的风景',
        'exact_response': NO_CONTEXT_ANSWER,
        'source_titles': [],
    },
]


def call_rag(query: str) -> dict:
    payload = json.dumps(
        {
            'messages': [{'role': 'user', 'content': query}],
            'use_rag': True,
        },
        ensure_ascii=False,
    ).encode('utf-8')
    request = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode('utf-8'))


def main() -> int:
    failures: list[str] = []

    for index, case in enumerate(CASES, start=1):
        result = call_rag(case['query'])
        response_text = result.get('response', '')
        source_titles = [item.get('title') for item in result.get('sources') or []]

        if 'exact_response' in case and response_text != case['exact_response']:
            failures.append(
                f'[{index}] {case["query"]}: expected exact response {case["exact_response"]!r}, got {response_text!r}'
            )

        for expected in case.get('must_include', []):
            if expected not in response_text:
                failures.append(f'[{index}] {case["query"]}: missing expected text {expected!r}')

        for unexpected in case.get('must_exclude', []):
            if unexpected in response_text:
                failures.append(f'[{index}] {case["query"]}: contained unexpected text {unexpected!r}')

        if source_titles != case.get('source_titles', source_titles):
            failures.append(
                f'[{index}] {case["query"]}: expected sources {case.get("source_titles")!r}, got {source_titles!r}'
            )

        status = 'OK' if not any(entry.startswith(f'[{index}]') for entry in failures) else 'FAIL'
        print(f'[{status}] {case["query"]}')
        print(f'  response: {response_text}')
        print(f'  sources: {source_titles}')

    if failures:
        print('\nFailures:')
        for failure in failures:
            print(f'- {failure}')
        return 1

    print('\nAll RAG probes passed.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f'Failed to reach backend: {exc}', file=sys.stderr)
        raise SystemExit(2)
