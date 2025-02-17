import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 본인의 API 키를 입력!
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

analysist = Agent(
    role="언어 분석가",
    goal="{input_text}에 대한 언어 및 문맥 분석 ",
    backstory="당신은 {input_text}에 대한 글을 분석하는 일을 하고 있습니다.\n"
              "글을 분석해 긍정적 언어와 부정적 언어를 탐지하고\n"
              "감정에 따라서 단어를 분류합니다.",
    allow_delegation=False,
    verbose=True
)

changer = Agent(
    role="전화 위복가",
    goal="{input_text}에 대한 의견을 포함한 글을 작성",
    backstory="당신은 {input_text}에 대한 글을 긍정적 표현으로 바꾸는 전화 위복가입니다.\n"
              "글의 부정적인 내용을 긍정적 표현을 사용해서 변경합니다.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="편집자",
    goal="주어진 글을 인터넷에서 유명한 장원영체로 편집",
    backstory="당신은 전화 위복가로부터 받은 글을 편집하는 편집자 입니다.",
    allow_delegation=False,
    verbose=True
)

analys = Task(
    description=(
        "{input_text}에 대한 감정을 분석하고, 긍정적인 감정과 "
        "부정적인 감정을 분류하기"
    ),
    expected_output="긍정적인 감정, 부정적인 감정",
    agent = analysist,
)

write = Task(
    description=(
        "긍정적인 감정을 활용해서, {input_text}의 글을"
        "긍적적으로 변환하기"
    ),
    expected_output="적절한 2~3문장의 글",
    agent = changer,
)

edit = Task(
    description=("적절한 2~3문장의 글에 마지막에 완전 럭키비키잖아🍀! 추가하기,그리고 친구한테 하는말처럼 편집하기"),
    expected_output="2~3문장 분량의 마크다운 형식의 글",
    agent=editor
)

crew = Crew(
    agents=[analysist, changer, editor],
    tasks=[analys, write, edit],
    verbose=2
)

result = crew.kickoff(inputs={"input_text": "오늘 구내식당 점심이 너무 맛없었어 ㅠㅠ"})

from IPython.display import Markdown
Markdown(result)
