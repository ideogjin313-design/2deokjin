# -*- coding: utf-8 -*-
"""Useful fragrance curation prototype."""

from __future__ import annotations

import csv
import base64
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from pathlib import Path
import time
from textwrap import dedent
from typing import Dict, List, Optional
import unicodedata

import numpy as np
import pickle
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    from rembg import remove
except ImportError:
    remove = None


st.set_page_config(
    page_title="유쏘풀 향수 큐레이션",
    page_icon="🌿",
    layout="wide",
)


STEP_PAGES = ["skin", "moisture", "temperature", "face", "loading", "result"]
STEP_LABELS = {
    "skin": "피부타입",
    "moisture": "수분감",
    "temperature": "온도",
    "face": "얼굴 진단",
    "loading": "분석 중",
    "result": "결과",
}

PAGE_PROGRESS_ALIAS = {
    "scent_loading": "result",
    "scent_result": "result",
}


SKIN_OPTIONS: List[Dict[str, object]] = [
    {
        "key": "지성",
        "icon": "✨",
        "title": "지성",
        "description": "기상 직후 번들거림이 심하고 유분감이 많이 느껴지는 편이에요.",
        "examples": "세안을 자주 하게 되고 피부가 답답하게 느껴질 때가 많아요.",
        "button": "지성으로 선택할게요",
        "filters": {
            "MolWt": "medium",
            "LogP": "medium",
            "TPSA": "medium",
            "BoilingPoint": "medium",
        },
    },
    {
        "key": "건성",
        "icon": "💧",
        "title": "건성",
        "description": "건조함과 당김이 자주 느껴지고 각질이 반복해서 올라오는 편이에요.",
        "examples": "세안 후 빠르게 건조해지고 피부가 갈라지는 느낌이 있을 수 있어요.",
        "button": "건성으로 선택할게요",
        "filters": {
            "MolWt": "high",
            "LogP": "high",
            "BoilingPoint": "high",
            "Complexity": "high",
            "TPSA": "low",
        },
    },
    {
        "key": "민감성",
        "icon": "🔴",
        "title": "민감성",
        "description": "자극, 홍조, 트러블이 잦고 진정이 오래 걸리는 편이에요.",
        "examples": "계절과 관계없이 피부가 쉽게 예민해지는 날이 많아요.",
        "button": "민감성으로 선택할게요",
        "filters": {
            "AromaticRings": "low",
            "Complexity": "low",
            "TPSA": "high",
        },
    },
    {
        "key": "중성",
        "icon": "🌿",
        "title": "잘 모르겠어요",
        "description": "특정 피부 타입으로 단정하기 어렵다면 중성 기준으로 진행할 수 있어요.",
        "examples": "이 경우에는 수분감과 피부 온도 정보를 중심으로 추천을 이어가요.",
        "button": "중성으로 진행할게요",
        "filters": {},
    },
]


MOISTURE_OPTIONS: List[Dict[str, object]] = [
    {
        "key": "보통",
        "icon": "💦",
        "title": "보통",
        "description": "극단적으로 건조하지도, 과하게 촉촉하지도 않은 균형형에 가까워요.",
        "examples": "계절 변화에 따라 조금 달라져도 기본 컨디션은 크게 흔들리지 않아요.",
        "button": "보통으로 반영할게요",
        "filters": {
            "MolWt": "low",
            "BoilingPoint": "low",
        },
    },
    {
        "key": "건조함",
        "icon": "🏜️",
        "title": "건조함",
        "description": "보습이 빨리 날아가고 메마름이 또렷하게 느껴지는 편이에요.",
        "examples": "시간이 지나면 피부 표면이 쉽게 푸석해지고 당김이 생길 수 있어요.",
        "button": "건조함으로 반영할게요",
        "filters": {
            "MolWt": "high",
            "BoilingPoint": "high",
        },
    },
]


TEMPERATURE_OPTIONS: List[Dict[str, object]] = [
    {
        "key": "높은 편",
        "icon": "🔥",
        "title": "높은 편",
        "description": "더위를 자주 느끼고 피부에 열감이 쉽게 오르는 편이에요.",
        "examples": "붉어짐이 잦거나 답답한 열감이 오래 남는 날이 있을 수 있어요.",
        "button": "높은 편으로 반영할게요",
        "filters": {
            "MolWt": "high",
            "BoilingPoint": "high",
            "RotatableBonds": "low",
        },
    },
    {
        "key": "낮은 편",
        "icon": "❄️",
        "title": "낮은 편",
        "description": "추위를 자주 느끼고 피부 열감을 잘 느끼지 못하는 편이에요.",
        "examples": "향이 천천히 퍼지거나 잔향이 오래 남는 편으로 느껴질 수 있어요.",
        "button": "낮은 편으로 반영할게요",
        "filters": {
            "MolWt": "low",
            "BoilingPoint": "low",
            "RotatableBonds": "high",
        },
    },
]


PERSONAL_COLOR_PROFILES: Dict[str, Dict[str, object]] = {
    "봄웜라이트": {
        "emoji": "🌸",
        "keywords": ["#풋풋함", "#달달함", "#포근함", "#사랑스러움", "#투명함"],
        "scent_feel": "연하고 파스텔톤의 색감 / 가볍고 달달하며 풋풋한 향 / 투명한 꽃잎이 연상되는 밝은 이미지",
        "description_lines": [
            "고명도 저채도의 밝고 부드러운 웜 파스텔 컬러로 이루어져 있어요.",
            "투명하고 풋풋한 느낌의 따뜻한 파스텔 톤이 은은하게 어우러져, 사랑스럽고 포근한 분위기가 특징이에요.",
        ],
    },
    "봄웜브라이트": {
        "emoji": "🌸",
        "keywords": ["#발랄함", "#화사함", "#톡톡튐", "#달콤함", "#생기"],
        "scent_feel": "선명하고 생기 넘치는 화려한 색감 / 달콤하면서 발랄하고 톡톡 튀는 향",
        "description_lines": [
            "고명도 고채도의 선명하고 생동감 넘치는 웜 컬러로 이루어져 있어요.",
            "달콤하면서도 발랄한 에너지로 생기 있는 분위기가 특징이에요.",
        ],
    },
    "여름쿨라이트": {
        "emoji": "🐋",
        "keywords": ["#청량함", "#우아함", "#맑음", "#부드러움", "#자연스러움"],
        "scent_feel": "화이트·핑크·보라 등 시원한 느낌의 컬러 / 플로럴 그린·아쿠아·바다",
        "description_lines": [
            "고명도 저채도의 맑고 투명한 쿨 파스텔 컬러로 이루어져 있어요.",
            "블루 베이스의 연하고 청량한 톤이 어우러져 자연스럽고 청순한 분위기가 특징이에요.",
        ],
    },
    "여름쿨브라이트": {
        "emoji": "🐋",
        "keywords": ["#산뜻함", "#상쾌함", "#깨끗함", "#청명함", "#싱그러움"],
        "scent_feel": "시원한 바다와 산뜻한 풀내음 / 깨끗하고 청랑한 향",
        "description_lines": [
            "고명도 고채도의 선명하고 시원한 쿨 컬러로 이루어져 있어요.",
            "흰끼가 가미된 쨍하고 쿨한 계열의 컬러가 조화를 이루어 산뜻하고 깨끗한 분위기가 특징이에요.",
        ],
    },
    "여름쿨뮤트": {
        "emoji": "🐋",
        "keywords": ["#은은함", "#차분함", "#세련됨", "#고급스러움", "#중성적"],
        "scent_feel": "회색이 가미된 차갑고 고급스러우며 우아인 컬러감 / 뿌린 듯 안 뿌린듯 은은한 향",
        "description_lines": [
            "중명도 저채도의 잿빛 도는 차분한 쿨 컬러로 이루어져 있어요.",
            "탁기 있는 컬러가 은은하게 스며들어 고급스럽고 세련된 분위기가 특징이에요.",
        ],
    },
    "가을웜소프트": {
        "emoji": "🍂",
        "keywords": ["#차분함", "#따뜻함", "#클래식함", "#고급스러움", "#내추럴"],
        "scent_feel": "차분하고 분위기 있으며 고급스러운 느낌 / 질고 풍부한 향 / 오리엔탈·우디·따뜻한 머스크",
        "description_lines": [
            "중명도 중채도의 부드럽고 따뜻한 웜 컬러로 이루어져 있어요.",
            "내추럴한 따뜻함이 녹아들어 차분하면서도 고급스러운 분위기가 특징이에요.",
        ],
    },
    "가을웜뮤트": {
        "emoji": "🍂",
        "keywords": ["#잔잔함", "#무게감", "#차분함", "#절제됨", "#분위기있음"],
        "scent_feel": "쿨과 웜의 중간 / 특정 향조의 특징 부각보다는 잔잔하고 무게감 있는 향",
        "description_lines": [
            "중명도 저채도의 잿빛 섞인 차분한 웜 컬러로 이루어져 있어요.",
            "탁기를 머금은 잔잔하고 무게감 있는 컬러감과 절제된 분위기 속 깊이 있는 느낌이 특징이에요.",
        ],
    },
    "가을웜딥": {
        "emoji": "🍂",
        "keywords": ["#성숙함", "#관능적", "#스모키", "#유혹적", "#고급스러움"],
        "scent_feel": "성숙함과 고급스러움 / 여성미가 느껴지는 클래식한 향 / 우디·스모키·앰버그리스·짙은 오리엔탈",
        "description_lines": [
            "저명도 중채도의 깊이 있고 짙은 웜 컬러로 이루어져 있어요.",
            "버건디, 딥브라운 계열의 묵직하고 성숙한 컬러감과 관능적이면서도 고급스러운 분위기가 특징이에요.",
        ],
    },
    "겨울쿨브라이트": {
        "emoji": "🌨",
        "keywords": ["#시원함", "#선명함", "#강인함", "#스포티함", "#도시적"],
        "scent_feel": "쨍한 파란색·체리 레드 등 통통 튀면서 차가운 컬러 / 커리어우먼 느낌 / 스파클링 와인·시프레 플로럴",
        "description_lines": [
            "고명도 고채도의 차갑고 선명한 비비드 컬러로 이루어져 있어요.",
            "블루 베이스의 쨍하고 강렬한 원색 쿨 계열 컬러와 도시적이고 강인한 세련미가 특징이에요.",
        ],
    },
    "겨울쿨딥": {
        "emoji": "🌨",
        "keywords": ["#묵직함", "#차가움", "#중후함", "#중성적", "#모던함"],
        "scent_feel": "묵직하면서 차가운 컬러 / 우디·오리엔탈 플로럴·메탈릭한 엣지·묵직한 머스크",
        "description_lines": [
            "저명도 중채도의 어둡고 딥한 쿨 컬러로 이루어져 있어요.",
            "네이비, 딥플럼 계열의 묵직하고 차가운 컬러감과 모던하고 중성적인 분위기가 특징이에요.",
        ],
    },
}


PERSONAL_COLOR_PALETTES: Dict[str, Dict[str, object]] = {
    "봄웜라이트": {
        "title": "봄웜라이트 최종 팔레트",
        "spec": "L*=86, C*=13, 웜",
        "colors": [
            {"hex": "#F1CFD4", "name": "연분홍레드", "theta": "0°", "a": "+13", "b": "+2", "c": "≈13.2"},
            {"hex": "#F4D0B9", "name": "살구복숭아", "theta": "45°", "a": "+9", "b": "+16", "c": "≈18.4"},
            {"hex": "#E7D6B0", "name": "버터옐로우", "theta": "90°", "a": "0", "b": "+35", "c": "35.0"},
            {"hex": "#CDDCC6", "name": "연민트그린", "theta": "135°", "a": "-9", "b": "+9", "c": "≈12.7"},
            {"hex": "#C4DEC6", "name": "파스텔그린", "theta": "180°", "a": "-13", "b": "0", "c": "13.0"},
            {"hex": "#C3DCD9", "name": "파스텔시안", "theta": "225°", "a": "-9", "b": "-5", "c": "≈10.3"},
            {"hex": "#E6D1E1", "name": "파스텔라벤더", "theta": "270°", "a": "+10", "b": "-10", "c": "≈14.1"},
            {"hex": "#F5CCE1", "name": "연라일락핑크", "theta": "315°", "a": "+18", "b": "-6", "c": "≈19.0"},
        ],
    },
    "봄웜브라이트": {
        "title": "봄웜브라이트 최종 팔레트",
        "spec": "L*=60~75, C*=35, 웜",
        "colors": [
            {"hex": "#DF6E78", "name": "선명코랄레드", "l": "60", "theta": "0°", "a": "+45", "b": "+15", "c": "≈47.4"},
            {"hex": "#EB957F", "name": "오렌지코랄", "l": "70", "theta": "45°", "a": "+30", "b": "+25", "c": "≈39.1"},
            {"hex": "#E2B06B", "name": "밝은옐로우", "l": "75", "theta": "90°", "a": "+10", "b": "+42", "c": "≈43.2"},
            {"hex": "#9AC48A", "name": "봄연두", "l": "75", "theta": "135°", "a": "-25", "b": "+25", "c": "≈35.4"},
            {"hex": "#64BC98", "name": "선명그린", "l": "70", "theta": "180°", "a": "-35", "b": "+10", "c": "≈36.4"},
            {"hex": "#37BAD7", "name": "밝은시안", "l": "70", "theta": "225°", "a": "-25", "b": "-25", "c": "≈35.4"},
            {"hex": "#65B2EA", "name": "선명블루", "l": "70", "theta": "270°", "a": "-8", "b": "-35", "c": "≈35.9"},
            {"hex": "#F889AD", "name": "작약핑크", "l": "70", "theta": "315°", "a": "+46", "b": "0", "c": "46.0"},
        ],
    },
    "여름쿨라이트": {
        "title": "여름쿨라이트 최종 팔레트",
        "spec": "L*=86, C*=13, 쿨",
        "colors": [
            {"hex": "#F0CFD7", "name": "로즈핑크", "theta": "0°", "a": "+13", "b": "0", "c": "13.0"},
            {"hex": "#ECD1CE", "name": "파스텔피치", "theta": "45°", "a": "+9", "b": "+5", "c": "≈10.3"},
            {"hex": "#E8D6AC", "name": "파스텔레몬", "theta": "90°", "a": "0", "b": "+23", "c": "23.0"},
            {"hex": "#C4DCD7", "name": "파스텔민트", "theta": "135°", "a": "-9", "b": "0", "c": "9.0"},
            {"hex": "#BBDED7", "name": "파스텔그린", "theta": "180°", "a": "-13", "b": "+6", "c": "≈14.3"},
            {"hex": "#E1D2E8", "name": "아이스시안", "theta": "225°", "a": "-8", "b": "-8", "c": "≈11.3"},
            {"hex": "#BEDCE6", "name": "파우더블루", "theta": "270°", "a": "0", "b": "-13", "c": "13.0"},
            {"hex": "#CAD8EF", "name": "라벤더", "theta": "315°", "a": "+9", "b": "-9", "c": "≈12.7"},
        ],
    },
    "여름쿨브라이트": {
        "title": "여름쿨브라이트 최종 팔레트",
        "spec": "L*=60~75, C*=35, 쿨",
        "colors": [
            {"hex": "#E97B9F", "name": "핫핑크레드", "l": "65", "theta": "0°", "a": "+46", "b": "0", "c": "46.0"},
            {"hex": "#D88884", "name": "마젠타오렌지", "l": "65", "theta": "45°", "a": "+30", "b": "+15", "c": "≈33.5"},
            {"hex": "#CCB781", "name": "레몬옐로우", "l": "75", "theta": "90°", "a": "0", "b": "+30", "c": "30.0"},
            {"hex": "#8AC5A9", "name": "쿨그린", "l": "75", "theta": "135°", "a": "-25", "b": "+8", "c": "≈26.2"},
            {"hex": "#52BCAA", "name": "선명쿨그린", "l": "70", "theta": "180°", "a": "-35", "b": "0", "c": "35.0"},
            {"hex": "#59C8D1", "name": "아이시시안", "l": "75", "theta": "225°", "a": "-29", "b": "-14", "c": "≈32.2"},
            {"hex": "#6294CE", "name": "코발트블루", "l": "60", "theta": "270°", "a": "0", "b": "-35", "c": "35.0"},
            {"hex": "#D86CA7", "name": "퓨어마젠타", "l": "60", "theta": "315°", "a": "+49", "b": "-12", "c": "≈50.4"},
        ],
    },
    "여름쿨뮤트": {
        "title": "여름쿨뮤트 최종 팔레트",
        "spec": "L*=50, C*=13, 쿨",
        "colors": [
            {"hex": "#8D6F77", "name": "더스티로즈", "theta": "0°", "a": "+13", "b": "0", "c": "13.0"},
            {"hex": "#8B7171", "name": "스모키피치", "theta": "45°", "a": "+10", "b": "+9", "c": "≈13.5"},
            {"hex": "#7A7770", "name": "스모키머스터드", "theta": "90°", "a": "0", "b": "+13", "c": "13.0"},
            {"hex": "#6A7B6E", "name": "스모키세이지", "theta": "135°", "a": "-9", "b": "+9", "c": "≈12.7"},
            {"hex": "#5D7D77", "name": "그레이그린", "theta": "180°", "a": "-13", "b": "0", "c": "13.0"},
            {"hex": "#627B86", "name": "스틸시안", "theta": "225°", "a": "-11", "b": "-9", "c": "≈14.2"},
            {"hex": "#6B788D", "name": "스틸블루", "theta": "270°", "a": "0", "b": "-13", "c": "13.0"},
            {"hex": "#8A6F86", "name": "모브", "theta": "315°", "a": "+15", "b": "-9", "c": "≈17.5"},
        ],
    },
    "가을웜소프트": {
        "title": "가을웜소프트 최종 팔레트",
        "spec": "L*=55, C*=20, 웜",
        "colors": [
            {"hex": "#A57784", "name": "소프트테라코타", "theta": "0°", "a": "+20", "b": "0", "c": "20.0"},
            {"hex": "#A37A6C", "name": "인디코랄", "theta": "45°", "a": "+14", "b": "+14", "c": "≈19.8"},
            {"hex": "#908261", "name": "허니옐로우", "theta": "90°", "a": "0", "b": "+20", "c": "20.0"},
            {"hex": "#748A6B", "name": "올리브그린", "theta": "135°", "a": "-14", "b": "+14", "c": "≈19.8"},
            {"hex": "#598D83", "name": "세이지그린", "theta": "180°", "a": "-20", "b": "0", "c": "20.0"},
            {"hex": "#548C9B", "name": "그레이쉬세이지", "theta": "225°", "a": "-14", "b": "-14", "c": "≈19.8"},
            {"hex": "#6F85A6", "name": "그레이쉬블루", "theta": "270°", "a": "0", "b": "-20", "c": "20.0"},
            {"hex": "#917C9C", "name": "더스티모브", "theta": "315°", "a": "+14", "b": "-14", "c": "≈19.8"},
        ],
    },
    "가을웜뮤트": {
        "title": "가을웜뮤트 최종 팔레트",
        "spec": "L*=44, C*=13, 웜",
        "colors": [
            {"hex": "#7D6068", "name": "머드브라운레드", "theta": "0°", "a": "+13", "b": "0", "c": "13.0"},
            {"hex": "#7C625A", "name": "오트밀베이지", "theta": "45°", "a": "+9", "b": "+9", "c": "≈12.7"},
            {"hex": "#706753", "name": "머스타드", "theta": "90°", "a": "0", "b": "+13", "c": "13.0"},
            {"hex": "#5F6C59", "name": "카키올리브", "theta": "135°", "a": "-9", "b": "+9", "c": "≈12.7"},
            {"hex": "#4F6E68", "name": "모스그린", "theta": "180°", "a": "-13", "b": "0", "c": "13.0"},
            {"hex": "#4D6D77", "name": "그레이쉬그린", "theta": "225°", "a": "-9", "b": "-9", "c": "≈12.7"},
            {"hex": "#5C697D", "name": "그레이쉬블루", "theta": "270°", "a": "0", "b": "-13", "c": "13.0"},
            {"hex": "#706477", "name": "탁한모브", "theta": "315°", "a": "+9", "b": "-9", "c": "≈12.7"},
        ],
    },
    "가을웜딥": {
        "title": "가을웜딥 최종 팔레트",
        "spec": "L*=28, C*≈25, 웜",
        "colors": [
            {"hex": "#653243", "name": "딥크림슨", "theta": "0°", "a": "+25", "b": "0", "c": "25.0"},
            {"hex": "#633627", "name": "딥번트오렌지", "theta": "45°", "a": "+18", "b": "+18", "c": "≈25.5"},
            {"hex": "#4E411B", "name": "딥골드브라운", "theta": "90°", "a": "0", "b": "+25", "c": "25.0"},
            {"hex": "#2E4925", "name": "딥올리브", "theta": "135°", "a": "-18", "b": "+18", "c": "≈25.5"},
            {"hex": "#3B461F", "name": "딥포레스트웜", "theta": "180°", "a": "-12", "b": "+22", "c": "≈25.1"},
            {"hex": "#204A2F", "name": "딥올리브웜", "theta": "225°", "a": "-22", "b": "+12", "c": "≈25.1"},
            {"hex": "#004B5D", "name": "딥슬레이트그린", "theta": "270°", "a": "-13", "b": "-16", "c": "≈20.6"},
            {"hex": "#63352F", "name": "딥로즈코랄", "theta": "315°", "a": "+20", "b": "+13", "c": "≈23.9"},
        ],
    },
    "겨울쿨브라이트": {
        "title": "겨울쿨브라이트 최종 팔레트",
        "spec": "L*=55, C*=35, 쿨",
        "colors": [
            {"hex": "#BC6B85", "name": "체리레드", "theta": "0°", "a": "+35", "b": "0", "c": "35.0"},
            {"hex": "#B8725A", "name": "코랄오렌지", "theta": "45°", "a": "+25", "b": "+25", "c": "≈35.4"},
            {"hex": "#978246", "name": "아이시레몬", "theta": "90°", "a": "0", "b": "+35", "c": "35.0"},
            {"hex": "#668E57", "name": "쿨라임", "theta": "135°", "a": "-25", "b": "+25", "c": "≈35.4"},
            {"hex": "#1F9383", "name": "쿨그린", "theta": "180°", "a": "-35", "b": "0", "c": "35.0"},
            {"hex": "#0091AE", "name": "아이시시안", "theta": "225°", "a": "-22", "b": "-24", "c": "≈32.5"},
            {"hex": "#5487C0", "name": "로열블루", "theta": "270°", "a": "0", "b": "-35", "c": "35.0"},
            {"hex": "#9A76AF", "name": "퓨어마젠타", "theta": "315°", "a": "+25", "b": "-25", "c": "≈35.4"},
        ],
    },
    "겨울쿨딥": {
        "title": "겨울쿨딥 최종 팔레트",
        "spec": "L*=32~42, C*=18~30, 쿨딥",
        "colors": [
            {"hex": "#743A34", "name": "버건디브라운", "theta": "15°", "a": "+18", "b": "+10", "c": "≈20.6"},
            {"hex": "#6A3F52", "name": "플럼와인", "theta": "345°", "a": "+18", "b": "-2", "c": "≈18.1"},
            {"hex": "#0B5B6B", "name": "딥틸", "theta": "210°", "a": "-18", "b": "-10", "c": "≈20.6"},
            {"hex": "#0E5974", "name": "페트롤블루", "theta": "240°", "a": "-14", "b": "-16", "c": "≈21.3"},
            {"hex": "#29527C", "name": "딥네이비", "theta": "265°", "a": "-4", "b": "-22", "c": "≈22.4"},
            {"hex": "#3E487B", "name": "잉크블루", "theta": "280°", "a": "+4", "b": "-20", "c": "≈20.4"},
            {"hex": "#4B4A80", "name": "딥아이리스", "theta": "295°", "a": "+8", "b": "-18", "c": "≈19.7"},
            {"hex": "#5E4472", "name": "블랙베리퍼플", "theta": "320°", "a": "+14", "b": "-12", "c": "≈18.4"},
        ],
    },
}




LABEL_BY_PERSONAL_COLOR = {
    "봄웜라이트": "가볍고 달달한 파스텔 플로럴",
    "봄웜브라이트": "생기 넘치는 스위트 프루티",
    "여름쿨라이트": "청량한 아쿠아 플로럴",
    "여름쿨브라이트": "깨끗한 아쿠아 그린",
    "여름쿨뮤트": "은은한 소프트 머스크",
    "가을웜소프트": "따뜻한 오리엔탈 우디",
    "가을웜뮤트": "잔잔한 웜 머스키 우디",
    "가을웜딥": "짙은 스모키 오리엔탈",
    "겨울쿨브라이트": "스파클링 시프레 플로럴",
    "겨울쿨딥": "딥 우디 머스크",
}


PRODUCT_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "가볍고 달달한 파스텔 플로럴": [
        {"name": "유쏘풀 페탈 블룸", "match": "적합도 100%", "notes": "화이트플라워, 피치, 소프트 머스크"},
        {"name": "유쏘풀 핑크 베일", "match": "적합도 72%", "notes": "피오니, 코튼, 스위트 프루트"},
    ],
    "생기 넘치는 스위트 프루티": [
        {"name": "유쏘풀 비비드 피치", "match": "적합도 100%", "notes": "피치, 시트러스, 플로럴"},
        {"name": "유쏘풀 써니 팝", "match": "적합도 70%", "notes": "애플, 자몽, 스위트 플라워"},
    ],
    "청량한 아쿠아 플로럴": [
        {"name": "유쏘풀 아쿠아 라이트", "match": "적합도 100%", "notes": "아쿠아, 라일락, 그린 플로럴"},
        {"name": "유쏘풀 블루 리프", "match": "적합도 68%", "notes": "워터리 노트, 플로럴, 허브"},
    ],
    "깨끗한 아쿠아 그린": [
        {"name": "유쏘풀 클린 쇼어", "match": "적합도 100%", "notes": "바다, 그린티, 풀내음"},
        {"name": "유쏘풀 프레시 웨이브", "match": "적합도 66%", "notes": "아쿠아, 허브, 시트러스"},
    ],
    "은은한 소프트 머스크": [
        {"name": "유쏘풀 뮤트 코튼", "match": "적합도 100%", "notes": "화이트 머스크, 아이리스, 코튼"},
        {"name": "유쏘풀 실버 베일", "match": "적합도 64%", "notes": "파우더, 머스크, 플로럴"},
    ],
    "따뜻한 오리엔탈 우디": [
        {"name": "유쏘풀 웜 앰버", "match": "적합도 100%", "notes": "오리엔탈, 우디, 따뜻한 머스크"},
        {"name": "유쏘풀 클래식 샌들", "match": "적합도 67%", "notes": "샌들우드, 앰버, 머스크"},
    ],
    "잔잔한 웜 머스키 우디": [
        {"name": "유쏘풀 드라이 우드", "match": "적합도 100%", "notes": "우디, 웜 머스크, 소프트 스파이스"},
        {"name": "유쏘풀 어텀 포그", "match": "적합도 63%", "notes": "머스크, 티우드, 잔향"},
    ],
    "짙은 스모키 오리엔탈": [
        {"name": "유쏘풀 딥 누아", "match": "적합도 100%", "notes": "우디, 스모키, 앰버그리스"},
        {"name": "유쏘풀 버건디 나잇", "match": "적합도 69%", "notes": "오리엔탈, 스모크, 레진"},
    ],
    "스파클링 시프레 플로럴": [
        {"name": "유쏘풀 시티 스파클", "match": "적합도 100%", "notes": "시프레, 스파클링 와인, 플로럴"},
        {"name": "유쏘풀 쿨 체리", "match": "적합도 65%", "notes": "체리 레드 무드, 시트러스, 플로럴"},
    ],
    "딥 우디 머스크": [
        {"name": "유쏘풀 모던 플럼", "match": "적합도 100%", "notes": "우디, 묵직한 머스크, 메탈릭 엣지"},
        {"name": "유쏘풀 나이트 네이비", "match": "적합도 66%", "notes": "오리엔탈 플로럴, 우디, 머스크"},
    ],
}


APP_DIR = Path(__file__).resolve().parent
MODEL_IMAGE_SIZE = 224

PERSONAL_COLOR_PALETTE_IMAGES = {
    "봄웜라이트": APP_DIR / "색상팔레트_봄웜라이트.png",
    "봄웜브라이트": APP_DIR / "색상팔레트_봄웜브라이트.png",
    "여름쿨라이트": APP_DIR / "색상팔레트_여름쿨라이트.png",
    "여름쿨브라이트": APP_DIR / "색상팔레트_여름쿨브라이트.png",
    "여름쿨뮤트": APP_DIR / "색상팔레트_여름쿨뮤트.png",
    "가을웜소프트": APP_DIR / "색상팔레트_가을웜소프트.png",
    "가을웜뮤트": APP_DIR / "색상팔레트_가을웜뮤트.png",
    "가을웜딥": APP_DIR / "색상팔레트_가을웜딥.png",
    "겨울쿨브라이트": APP_DIR / "색상팔레트_겨울쿨브라이트.png",
    "겨울쿨딥": APP_DIR / "색상팔레트_겨울쿨딥.png",
}

PERSONAL_COLOR_REPRESENTATIVE_IMAGES = {
    "봄웜라이트": APP_DIR / "대표이미지_봄웜라이트_가공.png",
    "봄웜브라이트": APP_DIR / "대표이미지_봄웜브라이트_가공.png",
    "여름쿨라이트": APP_DIR / "대표이미지_여름쿨라이트_가공.png",
    "여름쿨브라이트": APP_DIR / "대표이미지_여름쿨브라이트_가공.png",
    "여름쿨뮤트": APP_DIR / "대표이미지_여름쿨뮤트_가공.png",
    "가을웜소프트": APP_DIR / "대표이미지_가을웜소프트_가공.png",
    "가을웜뮤트": APP_DIR / "대표이미지_가을웜뮤트_가공.png",
    "가을웜딥": APP_DIR / "대표이미지_가을웜딥_가공.png",
    "겨울쿨브라이트": APP_DIR / "대표이미지_겨울쿨브라이트_가공.png",
    "겨울쿨딥": APP_DIR / "대표이미지_겨울쿨딥_가공.png",
}

PERSONAL_COLOR_MOOD_GUIDE = {
    "봄웜라이트": {
        "keywords": ["#풋풋함", "#달달함", "#사랑스러움", "#포근함", "#투명함"],
        "image_line": "투명하고 풋풋하며 사랑스러운 이미지",
    },
    "봄웜브라이트": {
        "keywords": ["#발랄함", "#생기발랄", "#화사함", "#에너지", "#선명함"],
        "image_line": "선명하고 생동감 넘치며 발랄한 이미지",
    },
    "여름쿨라이트": {
        "keywords": ["#청량함", "#우아함", "#맑음", "#부드러움", "#엘레강스"],
        "image_line": "맑고 청량하며 청순한 이미지",
    },
    "여름쿨브라이트": {
        "keywords": ["#산뜻함", "#깨끗함", "#청명함", "#상쾌함", "#시트러스"],
        "image_line": "산뜻하고 시원하며 싱그러운 이미지",
    },
    "여름쿨뮤트": {
        "keywords": ["#고급스러움", "#차분함", "#은은함", "#세련됨", "#도시적"],
        "image_line": "은은하고 차분하며 세련된 이미지",
    },
    "가을웜소프트": {
        "keywords": ["#클래식함", "#차분함", "#풍부함", "#따뜻함", "#오리엔탈"],
        "image_line": "차분하고 따뜻하며 고급스러운 이미지",
    },
    "가을웜뮤트": {
        "keywords": ["#잔잔함", "#무게감", "#차분함", "#깊이있음", "#절제됨"],
        "image_line": "잔잔하고 묵직하며 깊이 있는 이미지",
    },
    "가을웜딥": {
        "keywords": ["#성숙함", "#관능적", "#유혹적", "#고급스러움", "#강렬함"],
        "image_line": "성숙하고 관능적이며 강렬한 이미지",
    },
    "겨울쿨브라이트": {
        "keywords": ["#시원함", "#스포티함", "#선명함", "#도시적", "#강인함"],
        "image_line": "차갑고 선명하며 도시적인 이미지",
    },
    "겨울쿨딥": {
        "keywords": ["#묵직함", "#차가움", "#중후함", "#깊이있음", "#메탈릭"],
        "image_line": "차갑고 묵직하며 모던한 이미지",
    },
}

PERSONAL_COLOR_THEME = {
    "봄웜라이트": {"accent": "#d097a1", "button": "#d59ca8"},
    "봄웜브라이트": {"accent": "#f6657f", "button": "#f2576f"},
    "여름쿨라이트": {"accent": "#96badb", "button": "#8fb7dc"},
    "여름쿨브라이트": {"accent": "#2b7fcc", "button": "#247dca"},
    "여름쿨뮤트": {"accent": "#8a6f86", "button": "#9b7f9a"},
    "가을웜소프트": {"accent": "#d1bea5", "button": "#d4c0a2"},
    "가을웜뮤트": {"accent": "#926e5a", "button": "#a57a62"},
    "가을웜딥": {"accent": "#854023", "button": "#a44e24"},
    "겨울쿨브라이트": {"accent": "#c62a91", "button": "#ce2f97"},
    "겨울쿨딥": {"accent": "#171d37", "button": "#1d2648"},
}

SCENT_LABEL_GUIDE = {
    "Acidic": ("산미", "톡 쏘는 산미가 입안에 퍼지는 듯한 상큼한 향"),
    "Aldehydic": ("알데하이드", "막 씻은 린넨처럼 깨끗하게 퍼지는 투명한 향"),
    "Amber": ("앰버", "따뜻한 햇살 아래 포근하게 감싸는 달콤한 향"),
    "Animal": ("애니멀릭", "은은하게 남는 체온처럼 관능적인 깊은 향"),
    "Anisic": ("아니스", "달콤하고 스파이시한 향"),
    "Aromatic": ("아로마틱", "허브 정원 사이를 걷는 듯 풍부하게 퍼지는 향"),
    "Balsamic": ("발사믹", "부드럽고 달콤하게 감도는 향"),
    "Camphoraceous": ("캄퍼러스", "코를 시원하게 만드는 허브 느낌의 향"),
    "Citrus": ("시트러스", "갓 껍질을 벗긴 과일처럼 톡 터지는 상큼한 향"),
    "Earthy": ("어시(흙내음)", "비 온 뒤 흙내음을 머금은 자연 그대로의 향"),
    "Floral": ("플로럴", "꽃밭 위를 산책하는 듯 부드럽고 우아한 향"),
    "Food": ("푸디(음식)", "갓 만든 음식에서 느껴지는 고소하고 따뜻한 향"),
    "Fruity": ("프루티", "잘 익은 과일을 베어 문 듯 달콤한 향"),
    "Gourmand": ("구르망", "달콤한 디저트를 한 입 베어 문 듯한 향"),
    "Green": ("그린", "풀잎을 스칠 때 느껴지는 싱그럽고 생기 있는 향"),
    "Herbal": ("허브", "손으로 허브를 비빌 때 퍼지는 쌉싸름한 향"),
    "Honey": ("허니", "진한 꿀이 천천히 흐르는 듯 달콤한 향"),
    "Marine": ("마린", "바닷바람을 맞으며 느껴지는 시원하고 맑은 향"),
    "Minty": ("민트", "입안이 상쾌해지는 민트처럼 시원한 향"),
    "Musk": ("머스크", "부드러운 살결에서 느껴지는 포근한 잔향"),
    "Ozonic": ("오조닉", "맑은 공기를 깊게 들이마신 듯 깨끗한 향"),
    "Powdery": ("파우더리", "고운 파우더를 두른 듯 부드러운 향"),
    "Smoky": ("스모키", "장작불 옆에 앉아 있는 듯 은은하게 그을린 향"),
    "Spicy": ("스파이시", "따뜻한 향신료가 코끝을 자극하는 향"),
    "Sulfurous": ("설퍼러스", "강하게 퍼지는 자극적인 향"),
    "Tobacco": ("토바코", "말린 담배잎처럼 달콤하고 깊이 있는 향"),
    "Woody": ("우디", "나무 숲속의 차분하고 안정적인 향"),
}

SCENT_LABEL_FALLBACKS = {
    "Honey": ["Amber", "Fruity"],
    "Anisic": ["Spicy", "Amber"],
    "Aromatic": ["Herbal", "Green"],
    "Earthy": ["Woody", "Green"],
    "Food": ["Amber", "Fruity"],
    "Gourmand": ["Amber", "Fruity"],
    "Marine": ["Minty", "Green"],
    "Ozonic": ["Aldehydic", "Minty"],
}

PRODUCT_CATEGORY_COLUMN_MAP = {
    "MolWt": "MolWt_category",
    "LogP": "LogP_category",
    "TPSA": "TPSA_category",
    "BoilingPoint": "BP_pred_K_category",
    "RotatableBonds": "RotatableBonds_category",
    "AromaticRings": "AromaticRings_category",
    "Complexity": "Complexity_category",
}

SKIN_PROFILE_LABELS = {
    ("지성", "보통", "높은 편"): "번들·열감형 피부",
    ("지성", "보통", "낮은 편"): "비교적 안정적인 밸런스형 지성 피부",
    ("지성", "건조함", "높은 편"): "수부지 열감형 복합 피부",
    ("지성", "건조함", "낮은 편"): "수부지 피부",
    ("건성", "보통", "높은 편"): "예민 건성 피부",
    ("건성", "보통", "낮은 편"): "기본 건성 피부",
    ("건성", "건조함", "높은 편"): "극건성 피부",
    ("건성", "건조함", "낮은 편"): "저수분 건성 피부",
    ("민감성", "보통", "높은 편"): "민감한 홍조형 피부",
    ("민감성", "보통", "낮은 편"): "민감 피부",
    ("민감성", "건조함", "높은 편"): "예민·홍조 피부",
    ("민감성", "건조함", "낮은 편"): "쉽게 예민해지는 민감 피부",
    ("중성", "보통", "높은 편"): "열감이 도는 컨디션 피부",
    ("중성", "보통", "낮은 편"): "안정적인 건강 피부",
    ("중성", "건조함", "높은 편"): "수분과 열 관리가 필요한 피부",
    ("중성", "건조함", "낮은 편"): "수분 보충이 필요한 피부",
}

SKIN_PROFILE_TONE = {
    ("지성", "보통", "높은 편"): "유분감이 있고 열감도 느껴지는",
    ("지성", "보통", "낮은 편"): "비교적 안정적인 밸런스형",
    ("지성", "건조함", "높은 편"): "겉은 번들지만 속은 건조하고 열감이 있는",
    ("지성", "건조함", "낮은 편"): "유분감과 건조함이 함께 느껴지는",
    ("건성", "보통", "높은 편"): "건조하고 예민해지기 쉬운",
    ("건성", "보통", "낮은 편"): "건조함이 기본으로 느껴지는",
    ("건성", "건조함", "높은 편"): "매우 건조하고 열감까지 있는",
    ("건성", "건조함", "낮은 편"): "수분이 쉽게 부족해지는",
    ("민감성", "보통", "높은 편"): "예민하고 붉어지기 쉬운",
    ("민감성", "보통", "낮은 편"): "자극에 민감한",
    ("민감성", "건조함", "높은 편"): "건조하면서 예민하고 붉어지기 쉬운",
    ("민감성", "건조함", "낮은 편"): "쉽게 예민해지고 건조함이 느껴지는",
    ("중성", "보통", "높은 편"): "기본적으로는 안정적이지만 열감이 도는",
    ("중성", "보통", "낮은 편"): "비교적 안정적이고 편안한",
    ("중성", "건조함", "높은 편"): "수분 관리와 열감 관리가 함께 필요한",
    ("중성", "건조함", "낮은 편"): "수분 보충이 필요한",
}


def _find_first(pattern: str) -> Path:
    matches = sorted(APP_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Required file not found: {pattern}")
    return matches[0]


def _build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, num_classes))
    return model


def _build_mobilenet_v3_large(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    return model


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(in_features, num_classes))
    return model


def _load_state(model: nn.Module, pattern: str) -> nn.Module:
    state_dict = torch.load(_find_first(pattern), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_personal_color_assets() -> Dict[str, object]:
    classes = np.load(APP_DIR / "stacking_classes.npy", allow_pickle=True)

    with open(APP_DIR / "meta_model.pkl", "rb") as file:
        meta_model = pickle.load(file)
    if not hasattr(meta_model, "multi_class"):
        meta_model.multi_class = "multinomial"

    assets = {
        "classes": list(classes),
        "meta_model": meta_model,
        "resnet50": _load_state(_build_resnet50(len(classes)), "best_model_resnet50*.pth"),
        "mobilenetv3": _load_state(_build_mobilenet_v3_large(len(classes)), "best_model_mobilenetv3*.pth"),
        "efficientnet": _load_state(_build_efficientnet_b0(len(classes)), "best_model_efficientnet*.pth"),
        "transform": transforms.Compose(
            [
                transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    }
    return assets


def predict_personal_color_from_image(image_source) -> Dict[str, object]:
    assets = load_personal_color_assets()
    image = Image.open(image_source).convert("RGB")

    if remove is not None:
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        output_data = remove(image_buffer.getvalue())
        rembg_image = Image.open(BytesIO(output_data))
        if rembg_image.mode == "RGBA":
            background = Image.new("RGB", rembg_image.size, (0, 0, 0))
            background.paste(rembg_image, mask=rembg_image.split()[3])
            rembg_image = background
        else:
            rembg_image = rembg_image.convert("RGB")
    else:
        rembg_image = image

    image_tensor = assets["transform"](rembg_image).unsqueeze(0)

    base_probabilities: List[np.ndarray] = []
    with torch.inference_mode():
        for model_name in ("resnet50", "mobilenetv3", "efficientnet"):
            logits = assets[model_name](image_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            base_probabilities.append(probs)

    stacked_features = np.concatenate(base_probabilities, axis=1)
    meta_model = assets["meta_model"]
    prediction_index = int(meta_model.predict(stacked_features)[0])
    prediction_proba = meta_model.predict_proba(stacked_features)[0]
    predicted_label = assets["classes"][prediction_index]

    return {
        "label": predicted_label,
        "confidence": float(prediction_proba[prediction_index]),
        "probabilities": prediction_proba,
        "classes": assets["classes"],
        "rembg_image": rembg_image,
    }


def init_session() -> None:
    defaults = {
        "page": "home",
        "skin_type": None,
        "moisture_level": None,
        "temperature_level": None,
        "personal_color": "봄웜라이트",
        "uploaded_image": None,
        "rembg_image": None,
        "analysis_pending": False,
        "scent_analysis_pending": False,
        "prediction_confidence": None,
        "prediction_error": None,
        "recommended_labels": [],
        "recommended_label": None,
        "recommended_products": [],
        "priority_filters": OrderedDict(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def reset_all() -> None:
    st.session_state.page = "home"
    st.session_state.skin_type = None
    st.session_state.moisture_level = None
    st.session_state.temperature_level = None
    st.session_state.personal_color = "봄웜라이트"
    st.session_state.uploaded_image = None
    st.session_state.rembg_image = None
    st.session_state.analysis_pending = False
    st.session_state.scent_analysis_pending = False
    st.session_state.prediction_confidence = None
    st.session_state.prediction_error = None
    st.session_state.recommended_labels = []
    st.session_state.recommended_label = None
    st.session_state.recommended_products = []
    st.session_state.priority_filters = OrderedDict()


def current_step_index() -> int:
    current_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    return STEP_PAGES.index(current_page) if current_page in STEP_PAGES else -1


def get_option_by_key(options: List[Dict[str, object]], key: Optional[str]) -> Optional[Dict[str, object]]:
    for item in options:
        if item["key"] == key:
            return item
    return None


def merge_priority_filters() -> OrderedDict[str, str]:
    merged: OrderedDict[str, str] = OrderedDict()
    sources = [
        get_option_by_key(SKIN_OPTIONS, st.session_state.skin_type),
        get_option_by_key(MOISTURE_OPTIONS, st.session_state.moisture_level),
        get_option_by_key(TEMPERATURE_OPTIONS, st.session_state.temperature_level),
    ]
    for source in sources:
        if not source:
            continue
        for column, value in source["filters"].items():
            if column not in merged:
                merged[column] = value
    return merged


def explain_priority_conflicts() -> List[str]:
    messages: List[str] = []
    labels = [
        ("피부타입", get_option_by_key(SKIN_OPTIONS, st.session_state.skin_type)),
        ("수분감", get_option_by_key(MOISTURE_OPTIONS, st.session_state.moisture_level)),
        ("피부 온도", get_option_by_key(TEMPERATURE_OPTIONS, st.session_state.temperature_level)),
    ]
    for i, (first_name, first_item) in enumerate(labels):
        if not first_item:
            continue
        first_filters = first_item["filters"]
        for second_name, second_item in labels[i + 1 :]:
            if not second_item:
                continue
            for column, second_value in second_item["filters"].items():
                if column in first_filters and first_filters[column] != second_value:
                    messages.append(
                        f"{column}: `{first_name}` 우선 적용으로 `{first_filters[column]}`을 유지하고 `{second_name}`의 `{second_value}`는 제외했어요."
                    )
    return messages


def apply_global_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(229, 205, 179, 0.50), transparent 20%),
                radial-gradient(circle at 100% 100%, rgba(229, 205, 179, 0.35), transparent 18%),
                #f7f1ea;
            color: #2f2622;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 3.8rem;
            padding-bottom: 3rem;
        }
        .hero-card {
            background: rgba(255, 251, 247, 0.82);
            border: 1px solid #ead8c9;
            border-radius: 34px;
            padding: 2.4rem 2.8rem;
            box-shadow: 0 24px 60px rgba(199, 171, 145, 0.18);
        }
        .mini-card {
            background: rgba(255, 251, 247, 0.92);
            border: 1px solid #ead8c9;
            border-radius: 28px;
            padding: 1.3rem 1.2rem;
            min-height: 220px;
            box-shadow: 0 16px 40px rgba(199, 171, 145, 0.12);
        }
        .choice-card {
            background: rgba(255, 252, 248, 0.95);
            border: 1px solid #ead8c9;
            border-radius: 30px;
            padding: 1.9rem 1.8rem 1.5rem 1.8rem;
            box-shadow: 0 18px 40px rgba(199, 171, 145, 0.10);
            margin-bottom: 0.9rem;
        }
        .choice-title {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.8rem;
            font-weight: 800;
            color: #2f2622;
        }
        .icon-badge {
            width: 52px;
            height: 52px;
            border-radius: 999px;
            border: 1px solid #dcc6b4;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #fffaf6;
            font-size: 1.6rem;
        }
        .result-shell {
            border: 1px solid #e4cdbd;
            background: rgba(255, 250, 245, 0.92);
            border-radius: 34px;
            padding: 1.4rem;
        }
        .output-head {
            background: #a98365;
            color: white;
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 1.2rem;
        }
        .result-box {
            background: #fffdfb;
            border: 3px solid #8d8d8d;
            border-radius: 12px;
            min-height: 84px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 1.2rem;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
            color: #221f1d;
            margin-bottom: 0.9rem;
        }
        .result-side-note {
            min-height: 84px;
            display: flex;
            align-items: center;
            font-size: 1.15rem;
            font-weight: 700;
            color: #1f1a18;
            margin-bottom: 0.9rem;
            padding-left: 0.7rem;
            line-height: 1.5;
        }
        .product-card {
            background: rgba(255, 252, 248, 0.95);
            border: 1px solid #ead8c9;
            border-radius: 28px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }
        .product-chip {
            display: inline-block;
            margin-top: 0.8rem;
            border: 1px solid #e5cdb8;
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
            color: #8b674d;
            background: #fffdfb;
            font-size: 0.92rem;
        }
        div.stButton > button {
            width: 100%;
            border-radius: 999px;
            min-height: 58px;
            border: 1px solid #d8bca5;
            background: #fffaf6;
            color: #332823;
            font-size: 1rem;
            font-weight: 600;
        }
        div.stButton > button[kind="primary"] {
            background: #a97757;
            color: white;
            border: none;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            color: #6f645b !important;
            background: rgba(255, 250, 245, 0.92);
            border: 1px solid #ead8c9;
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
        }
        .stTabs [aria-selected="true"] {
            color: #a06f50 !important;
            background: #fff6ef;
            border-color: #d8bca5;
        }
        .upload-preview {
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .upload-status {
            background: linear-gradient(135deg, rgba(236, 247, 226, 0.95), rgba(223, 242, 217, 0.92));
            border: 1px solid #c8e3bc;
            border-radius: 22px;
            padding: 1rem 1.1rem;
            color: #45704a;
            font-weight: 700;
            box-shadow: 0 14px 30px rgba(160, 196, 145, 0.14);
            margin-top: 0.8rem;
            margin-bottom: 1rem;
        }
        .palette-card {
            background: rgba(255, 252, 248, 0.96);
            border: 1px solid #ead8c9;
            border-radius: 30px;
            padding: 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 40px rgba(199, 171, 145, 0.10);
        }
        .palette-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .swatch-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 0.6rem;
            margin-bottom: 1rem;
        }
        .swatch-circle {
            width: 92px;
            height: 92px;
            border-radius: 999px;
            border: 4px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 10px 24px rgba(122, 109, 100, 0.18);
        }
        .swatch-label {
            font-size: 0.95rem;
            font-weight: 800;
            color: #2f2622;
            line-height: 1.35;
        }
        .season-card {
            max-width: 640px;
            margin: 0 auto 1.3rem auto;
            background: rgba(255, 253, 250, 0.97);
            border-radius: 40px;
            padding: 1.9rem 1.8rem 1.7rem 1.8rem;
            box-shadow: 0 24px 55px rgba(199, 171, 145, 0.12);
            text-align: center;
        }
        .season-avatar {
            width: 168px;
            height: 168px;
            border-radius: 999px;
            object-fit: cover;
            display: block;
            margin: 0 auto 1.15rem auto;
            box-shadow: 0 20px 34px rgba(122, 109, 100, 0.16);
        }
        .season-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
        }
        .season-pill-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.55rem;
            margin-bottom: 1.1rem;
        }
        .season-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.42rem 1rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #fff;
            min-width: 108px;
            box-shadow: 0 10px 18px rgba(122, 109, 100, 0.12);
        }
        .season-body {
            font-size: 1rem;
            line-height: 1.8;
            color: #554843;
            margin-top: 0.95rem;
        }
        .season-divider {
            width: 78%;
            height: 1px;
            margin: 1.45rem auto 1.2rem auto;
            background: currentColor;
            opacity: 0.35;
        }
        .season-subtitle {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .season-swatch-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.95rem 0.6rem;
            max-width: 420px;
            margin: 0 auto 1.35rem auto;
        }
        .season-swatch-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.45rem;
        }
        .season-swatch-circle {
            width: 68px;
            height: 68px;
            border-radius: 999px;
            border: 4px solid rgba(255, 255, 255, 0.96);
            box-shadow: 0 10px 20px rgba(122, 109, 100, 0.14);
        }
        .season-swatch-name {
            font-size: 0.8rem;
            line-height: 1.35;
            font-weight: 700;
            color: #5f524c;
            word-break: keep-all;
        }
        .season-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 270px;
            border-radius: 999px;
            padding: 0.92rem 1.4rem;
            color: white;
            font-size: 1.08rem;
            font-weight: 800;
            box-shadow: 0 14px 24px rgba(122, 109, 100, 0.18);
        }
        .fragrance-card {
            max-width: 760px;
            margin: 0 auto 1.25rem auto;
            background: rgba(255, 253, 250, 0.97);
            border: 1px solid #ead8c9;
            border-radius: 38px;
            padding: 1.8rem 1.8rem 1.5rem 1.8rem;
            box-shadow: 0 20px 46px rgba(199, 171, 145, 0.11);
        }
        .fragrance-title {
            text-align: center;
            font-size: 1.9rem;
            font-weight: 800;
            margin-bottom: 1.4rem;
        }
        .fragrance-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .fragrance-item {
            text-align: center;
        }
        .fragrance-orb {
            width: 132px;
            height: 132px;
            border-radius: 999px;
            margin: 0 auto 0.9rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 0;
            overflow: hidden;
            color: white;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.4;
            box-shadow: 0 16px 28px rgba(122, 109, 100, 0.16);
        }
        .fragrance-name {
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .fragrance-desc {
            font-size: 0.95rem;
            line-height: 1.7;
            color: #6f645b;
        }
        .fragrance-summary {
            text-align: center;
            font-size: 1rem;
            line-height: 1.9;
            color: #2f2622;
            font-weight: 700;
            margin: 1rem auto 1.2rem auto;
            max-width: 620px;
        }
        .featured-product-card {
            max-width: 560px;
            margin: 0 auto;
            border-radius: 28px;
            padding: 1.2rem 1.3rem 1.25rem 1.3rem;
            background: rgba(255,255,255,0.78);
            border: 1px solid #ead8c9;
            text-align: center;
        }
        .featured-product-image {
            width: 100%;
            max-height: 460px;
            object-fit: cover;
            display: block;
            border-radius: 24px;
            margin-bottom: 0.7rem;
            box-shadow: 0 16px 34px rgba(122, 109, 100, 0.14);
        }
        .featured-product-note {
            font-size: 0.92rem;
            line-height: 1.75;
            color: #403631;
            font-weight: 700;
        }
        .result-featured-product-card {
            max-width: 560px;
            margin: 1.4rem auto 0 auto;
            border-radius: 28px;
            padding: 0.95rem 1.1rem 0.95rem 1.1rem;
            background: rgba(255,255,255,0.82);
            border: 1px solid #ead8c9;
            text-align: center;
            box-shadow: 0 18px 40px rgba(122, 109, 100, 0.1);
        }
        .product-image {
            width: 126px;
            height: 168px;
            object-fit: cover;
            border-radius: 22px;
            box-shadow: 0 14px 28px rgba(122, 109, 100, 0.12);
            flex-shrink: 0;
        }
        .loading-panel {
            max-width: 370px;
            min-height: 520px;
            margin: 1.4rem auto 0 auto;
            background: rgba(255, 253, 250, 0.97);
            border-radius: 38px;
            border: 1.5px solid #e2d2c6;
            box-shadow: 0 24px 56px rgba(199, 171, 145, 0.12);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem 1.8rem;
        }
        .loading-ring {
            position: relative;
            width: 104px;
            height: 104px;
            margin-bottom: 1.35rem;
        }
        .loading-dot {
            position: absolute;
            width: 17px;
            height: 17px;
            border-radius: 999px;
            background: #d8dfe5;
            top: 50%;
            left: 50%;
            margin: -8.5px;
            transform-origin: 0 0;
            animation: loadingFade 1.15s linear infinite;
        }
        @keyframes loadingFade {
            0% { opacity: 1; background: #55626e; }
            100% { opacity: 0.18; background: #d8dfe5; }
        }
        @media (max-width: 768px) {
            .season-card,
            .fragrance-card {
                padding-left: 1.1rem;
                padding-right: 1.1rem;
            }
            .season-swatch-grid,
            .fragrance-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .season-cta {
                min-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return
    current = current_step_index()
    cols = st.columns(len(STEP_PAGES))
    for idx, (col, page) in enumerate(zip(cols, STEP_PAGES), start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="
                        width:50px;
                        height:50px;
                        margin:0 auto 0.55rem auto;
                        border-radius:999px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:{bubble_bg};
                        color:white;
                        font-weight:800;
                        font-size:1.25rem;
                    ">{bubble_text}</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{label_color};">{STEP_LABELS[page]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="color:#a06f50; font-weight:800; letter-spacing:0.16em; font-size:0.9rem;">{kicker}</div>
            <div style="font-size:2.8rem; line-height:1.2; font-weight:800; margin-top:1rem; color:#2f2622; white-space:pre-line;">{title}</div>
            <div style="font-size:1.08rem; line-height:1.8; color:#7a6d64; margin-top:1.25rem; white-space:pre-line;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    render_hero(
        "유쏘풀 향 추천",
        "피부 상태와 퍼스널 컬러를 반영한\n유쏘풀 향수 큐레이션",
        "피부 상태를 먼저 확인하고 얼굴 사진으로 퍼스널 컬러를 분석해요.\n그 결과를 바탕으로 어울리는 향과 유쏘풀 제품까지 차례로 추천해드릴게요.",
    )
    st.markdown("")
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("🧴", "피부 상태 확인", "피부 타입, 수분감, 피부 온도를 차례로 체크해 현재 피부 컨디션을 먼저 파악해요."),
        ("🌸", "퍼스널 컬러 분석", "얼굴 사진을 바탕으로 퍼스널 컬러를 분석하고 결과 카드를 보여드려요."),
        ("🏷️", "어울리는 향 추천", "퍼스널 컬러 결과와 피부 데이터를 함께 반영해 어울리는 향을 골라드려요."),
        ("🫧", "향수 제품 추천", "추천된 어울리는 향 계열과 피부 특성을 바탕으로 유쏘풀 제품까지 이어서 추천해드려요."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-size:1.55rem; font-weight:800; margin-top:1rem; color:#2f2622;">{title}</div>
                    <div style="font-size:1rem; line-height:1.7; color:#7a6d64; margin-top:0.9rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("")
    left, center, right = st.columns([1.3, 1.5, 1.3])
    with center:
        if st.button("향수 추천 시작하기 →", type="primary", use_container_width=True):
            go_to("skin")


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    render_hero(kicker, title, description)
    st.markdown("")
    for item in options:
        st.markdown(
            f"""
            <div class="choice-card">
                <div class="choice-title">
                    <span class="icon-badge">{item['icon']}</span>
                    <span>{item['title']}</span>
                </div>
                <div style="font-size:1rem; line-height:1.9; color:#6f645b; margin-top:1rem;">{item['description']}</div>
                <div style="font-size:0.98rem; line-height:1.8; color:#8a776a; margin-top:0.6rem;">{item['examples']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(item["button"], key=f"{state_key}_{item['key']}", use_container_width=True):
            st.session_state[state_key] = item["key"]
            go_to(next_page)
    left, right = st.columns(2)
    with left:
        if st.button("← 이전", key=f"prev_{state_key}", use_container_width=True):
            go_to(prev_page)
    with right:
        selected = st.session_state[state_key]
        st.button(
            f"{selected} 선택됨" if selected else "항목을 선택해 주세요",
            disabled=True,
            key=f"current_{state_key}",
            use_container_width=True,
        )


def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()
    render_step_progress()
    render_hero(
        "얼굴 사진 분석",
        "얼굴 사진으로\n퍼스널 컬러를 분석할게요",
        "사진을 올리면 배경을 정리한 뒤 퍼스널 컬러를 분석해요.\n분석이 끝나면 결과 카드와 향 추천 화면으로 이어집니다.",
    )
    st.markdown("")
    left_space, center_area, right_space = st.columns([0.9, 1.6, 0.9])
    with center_area:
        tab1, tab2 = st.tabs(["📁 사진 업로드", "📷 카메라로 찍기"])
        with tab1:
            uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        with tab2:
            camera_photo = st.camera_input("카메라 촬영")
            if camera_photo is not None:
                st.session_state.uploaded_image = camera_photo
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        if st.session_state.uploaded_image is not None:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="업로드된 사진", width=420)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="upload-status">사진 업로드가 완료됐어요. 결과 보기를 누르면 분석을 시작할게요.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        if st.button("← 이전", key="face_prev", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def derive_label_from_personal_color(personal_color: str) -> str:
    return LABEL_BY_PERSONAL_COLOR[personal_color]


@st.cache_data(show_spinner=False)
def load_personal_color_scent_labels() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with open(APP_DIR / "personal_color_scent_final_top6.csv", "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels = [
                row.get("향라벨_top1", "").strip(),
                row.get("향라벨_top2", "").strip(),
                row.get("향라벨_top3", "").strip(),
            ]
            mapping[row["퍼스널컬러"].strip()] = [label for label in labels if label]
    return mapping


@st.cache_data(show_spinner=False)
def load_youssoful_products() -> List[Dict[str, str]]:
    with open(APP_DIR / "youssoful_DB.csv", "r", encoding="cp949", newline="") as file:
        return list(csv.DictReader(file))


def normalize_filter_value(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "medium":
        return "mid"
    return normalized


def expand_scent_labels(labels: List[str]) -> List[str]:
    expanded: List[str] = []
    for label in labels:
        if label not in expanded:
            expanded.append(label)
        for fallback in SCENT_LABEL_FALLBACKS.get(label, []):
            if fallback not in expanded:
                expanded.append(fallback)
    return expanded


def recommend_youssoful_products(
    personal_color: str,
    recommended_labels: List[str],
    priority_filters: OrderedDict[str, str],
    top_n: int = 3,
) -> List[Dict[str, str]]:
    rows = load_youssoful_products()
    expanded_labels = expand_scent_labels(recommended_labels)
    label_rank = {label: index for index, label in enumerate(expanded_labels)}
    original_label_set = set(recommended_labels)

    scored_products: Dict[str, Dict[str, object]] = {}
    for row in rows:
        row_label = (row.get("Label") or "").strip()
        if expanded_labels and row_label not in label_rank:
            continue

        score = 0
        reasons: List[str] = []

        if row_label in label_rank:
            rank = label_rank[row_label]
            if row_label in original_label_set:
                score += 120 - (rank * 18)
                label_name = SCENT_LABEL_GUIDE.get(row_label, (row_label, ""))[0]
                reasons.append(f"퍼스널 컬러 어울리는 향 {label_name}")
            else:
                score += 78 - (rank * 8)
                label_name = SCENT_LABEL_GUIDE.get(row_label, (row_label, ""))[0]
                reasons.append(f"유사한 향 계열 {label_name}")

        matched_filters = 0
        for source_column, preferred_value in priority_filters.items():
            category_column = PRODUCT_CATEGORY_COLUMN_MAP.get(source_column)
            if not category_column:
                continue
            actual_value = normalize_filter_value(row.get(category_column, ""))
            if actual_value == normalize_filter_value(preferred_value):
                score += 14
                matched_filters += 1

        if matched_filters:
            reasons.append(f"피부 조건 {matched_filters}개 일치")

        product_key = (row.get("product_name") or "").strip()
        if not product_key:
            continue

        product_data = {
            "name": product_key,
            "match": " · ".join(reasons) if reasons else "퍼스널 컬러와 피부 조건을 반영한 추천",
            "notes": (row.get("product_label_text") or "").strip() or (row.get("Label") or "").strip(),
            "label": row_label,
            "score": score,
        }

        existing = scored_products.get(product_key)
        if existing is None or product_data["score"] > existing["score"]:
            scored_products[product_key] = product_data

    ranked = sorted(scored_products.values(), key=lambda item: (-item["score"], item["name"]))
    if ranked:
        return ranked[:top_n]

    fallback_products: List[Dict[str, str]] = []
    for row in rows[:top_n]:
        fallback_products.append(
            {
                "name": (row.get("product_name") or "").strip(),
                "match": "현재 데이터 기준 기본 추천",
                "notes": (row.get("product_label_text") or "").strip() or (row.get("Label") or "").strip(),
                "label": (row.get("Label") or "").strip(),
                "score": 0,
            }
        )
    return fallback_products


def run_recommendation() -> None:
    st.session_state.priority_filters = merge_priority_filters()
    if st.session_state.uploaded_image is not None:
        try:
            prediction = predict_personal_color_from_image(st.session_state.uploaded_image)
            st.session_state.personal_color = prediction["label"]
            st.session_state.rembg_image = prediction["rembg_image"]
            st.session_state.prediction_confidence = prediction["confidence"]
            st.session_state.prediction_error = None
        except Exception as error:
            st.session_state.rembg_image = None
            st.session_state.prediction_confidence = None
            st.session_state.prediction_error = str(error)
    scent_label_mapping = load_personal_color_scent_labels()
    recommended_labels = scent_label_mapping.get(st.session_state.personal_color, [])
    st.session_state.recommended_labels = recommended_labels
    st.session_state.recommended_label = derive_label_from_personal_color(st.session_state.personal_color)
    st.session_state.recommended_products = recommend_youssoful_products(
        st.session_state.personal_color,
        recommended_labels,
        st.session_state.priority_filters,
    )


def render_loading_page() -> None:
    if st.session_state.uploaded_image is None:
        go_to("face")
        return

    st.markdown(
        """
        <style>
        .stButton,
        .stTabs,
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"],
        [data-testid="stImage"],
        .upload-preview,
        .upload-status {
            display:none !important;
        }
        .block-container {
            padding-top: 5.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1rem; line-height:1.9; color:#675b54;">
                업로드한 사진을 바탕으로 퍼스널 컬러를 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.analysis_pending:
        time.sleep(1.0)
        run_recommendation()
        st.session_state.analysis_pending = False
    st.session_state.page = "result"
    st.rerun()


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(47, 38, 34, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def image_to_base64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def normalize_asset_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def get_scent_label_image_path(label: str) -> Optional[Path]:
    target = unicodedata.normalize("NFC", label).lower()
    for path in APP_DIR.glob("*.png"):
        normalized_name = unicodedata.normalize("NFC", path.name).lower()
        if target in normalized_name:
            return path
    return None


PRODUCT_IMAGE_KEY_ALIASES = {
    "dimancheaparis": ["dimancheaparis", "dimanche_a_paris"],
    "saintmoritz": ["saintmoritz", "saint_moritz", "saintmorits", "saint_morits"],
    "jetaime": ["jetaime", "jetaime", "jetaime", "jetaime"],
    "berrynoir": ["berrynoir", "berry_noir"],
    "berrysexy": ["berrysexy", "berry_sexy"],
    "cocohawaii": ["cocohawaii", "coco_hawaii"],
    "cotedazur": ["cotedazur", "cote_dazur", "cotedazur"],
    "marysgarden": ["marysgarden", "mary_garden", "marysgarden"],
    "staremesto": ["staremesto", "stare_mesto"],
    "divingcefaluspiaggia": ["divingcefaluspiaggia", "diving_cefalu_spiaggia"],
    "boroughlondon": ["boroughlondon", "borough_london"],
    "emporiacotton": ["emporiacotton", "emporia_cotton"],
    "milanfever": ["milanfever", "milan_fever"],
    "pugliaostuni": ["pugliaostuni", "puglia_ostuni"],
    "santalsequoia": ["santalsequoia", "santal_sequoia"],
    "kingsman": ["kingsman"],
    "daiquiri": ["daiquiri"],
    "hyperion": ["hyperion"],
    "seychelles": ["seychelles"],
    "biove": ["biove"],
    "floralia": ["floralia"],
    "fiine": ["fiine"],
    "lourdes": ["lourdes"],
    "burjroyal": ["burjroyal", "burj_royal"],
}

PRODUCT_IMAGE_FILE_MAP = {
    "berrynoir": "image (15).png",
    "berrysexy": "image (16).png",
    "pugliaostuni": "image (17).png",
    "divingcefaluspiaggia": "image (18).png",
    "santalsequoia": "image (19).png",
    "emporiacotton": "image (20).png",
    "floralia": "image (21).png",
    "milanfever": "image (22).png",
    "saintmoritz": "image (23).png",
    "jetaime": "image (24).png",
    "marysgarden": "image (25).png",
    "staremesto": "image (26).png",
    "cotedazur": "image (27).png",
    "lourdes": "image (28).png",
    "hyperion": "image (29).png",
    "kingsman": "image (11).png",
    "daiquiri": "image (32).png",
    "cocohawaii": "image (33).png",
    "burjroyal": "image (34).png",
    "seychelles": "image (12).png",
    "boroughlondon": "image (13).png",
    "fiine": "image (14).png",
}


def get_product_image_path(product_name: str) -> Optional[Path]:
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    product_key = normalize_asset_key(product_name)
    candidate_keys = [product_key]
    candidate_keys.extend(PRODUCT_IMAGE_KEY_ALIASES.get(product_key, []))

    for candidate_key in candidate_keys:
        mapped_filename = PRODUCT_IMAGE_FILE_MAP.get(candidate_key)
        if mapped_filename:
            mapped_path = APP_DIR / mapped_filename
            if mapped_path.is_file():
                return mapped_path

    for path in APP_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        normalized_name = normalize_asset_key(path.stem)
        if any(candidate_key and candidate_key in normalized_name for candidate_key in candidate_keys):
            return path
    return None


def build_product_image_markup(product_name: str, css_class: str = "product-image") -> str:
    image_path = get_product_image_path(product_name)
    image_data = image_to_base64(image_path) if image_path else None
    if not image_data or not image_path:
        return ""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix == "jpg" else suffix
    return (
        f'<img class="{css_class}" src="data:image/{mime};base64,{image_data}" '
        f'alt="{product_name}" />'
    )


def get_palette_caption(palette: Dict[str, object], profile: Dict[str, object]) -> str:
    color_names = [color.get("name", "").strip() for color in palette.get("colors", []) if color.get("name", "").strip()]
    if color_names:
        return " / ".join(color_names[:4])
    keywords = [keyword.replace("#", "") for keyword in profile.get("keywords", []) if keyword]
    return " / ".join(keywords[:4])


def render_signature_palette_card(personal_color: str) -> None:
    profile = PERSONAL_COLOR_PROFILES.get(personal_color)
    palette = PERSONAL_COLOR_PALETTES.get(personal_color)
    palette_image_path = PERSONAL_COLOR_PALETTE_IMAGES.get(personal_color)
    mood = PERSONAL_COLOR_MOOD_GUIDE.get(personal_color)
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_path = PERSONAL_COLOR_REPRESENTATIVE_IMAGES.get(personal_color)

    if not profile or not palette or not mood:
        return

    image_data = image_to_base64(image_path) if image_path else None
    image_markup = ""
    if image_data:
        image_markup = (
            f'<img class="season-avatar" src="data:image/png;base64,{image_data}" alt="{personal_color} 대표 이미지" />'
        )

    keyword_pills = "".join(
        f'<span class="season-pill" style="background:{theme["accent"]};">{keyword}</span>'
        for keyword in profile["keywords"]
    )
    colors = palette["colors"]
    palette_caption = get_palette_caption(palette, profile)
    swatch_markup = "".join(
        f"""
        <div class="season-swatch-item">
            <div class="season-swatch-circle" style="background:{color['hex']};"></div>
            <div class="season-swatch-name">{color['name']}</div>
        </div>
        """
        for color in colors
    )
    palette_markup = f'<div class="season-swatch-grid">{swatch_markup}</div>'
    if not colors and palette_image_path and palette_image_path.exists():
        palette_image_data = image_to_base64(palette_image_path)
        if palette_image_data:
            palette_markup = (
                f'<div style="max-width:420px; margin:0 auto 1.35rem auto;">'
                f'<img src="data:image/png;base64,{palette_image_data}" alt="{personal_color} 팔레트" '
                f'style="width:100%; display:block; border-radius:28px;" />'
                f"</div>"
            )
    description_lines = "<br>".join(profile["description_lines"])

    st.markdown(
        f"""
        <div class="season-card" style="
            border:1.6px solid {hex_to_rgba(theme['accent'], 0.72)};
            box-shadow: 0 24px 55px {hex_to_rgba(theme['accent'], 0.16)};
        ">
            {image_markup}
            <div class="season-title" style="color:{theme['accent']};">{personal_color}</div>
            <div class="season-pill-row">{keyword_pills}</div>
            <div class="season-body">{description_lines}</div>
            <div class="season-divider" style="color:{theme['accent']};"></div>
            <div class="season-subtitle" style="color:{theme['accent']};">어울리는 색상</div>
            {palette_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_featured_product_card(personal_color: str, featured_product: Dict[str, str]) -> None:
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_markup = build_product_image_markup(featured_product["name"], "featured-product-image")
    skin_profile = build_skin_profile_phrase()
    skin_recommendation_line = build_skin_profile_recommendation_line()
    st.markdown(
        f"""
        <div class="result-featured-product-card" style="border-color:{hex_to_rgba(theme['accent'], 0.24)};">
            <div style="font-size:0.92rem; font-weight:700; color:{theme['accent']}; margin-bottom:0.55rem;">유쏘풀 추천 향수</div>
            {image_markup}
            <div style="font-size:1.42rem; font-weight:800; color:#2f2622; margin:0.2rem 0 0.55rem 0;">{featured_product['name']}</div>
            <div class="featured-product-note">{featured_product['notes']}</div>
            <div class="featured-product-note" style="margin-top:0.65rem;">{skin_profile}</div>
            <div class="featured-product-note" style="margin-top:0.35rem;">{skin_recommendation_line}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fragrance_section(personal_color: str, recommended_labels: List[str], products: List[Dict[str, str]]) -> None:
    mood = PERSONAL_COLOR_MOOD_GUIDE.get(personal_color)
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    skin_profile = build_skin_profile_phrase()
    skin_recommendation_line = build_skin_profile_recommendation_line()
    if not recommended_labels:
        return

    item_markup = ""
    for label in recommended_labels[:3]:
        label_name, label_desc = SCENT_LABEL_GUIDE.get(label, (label, "설명이 아직 준비되지 않았어요."))
        image_path = get_scent_label_image_path(label)
        image_data = image_to_base64(image_path) if image_path else None
        orb_inner = label_name
        orb_style = f"background: radial-gradient(circle at 30% 25%, {hex_to_rgba(theme['accent'], 0.78)}, {theme['button']});"
        if image_data:
            orb_style = "background: transparent; box-shadow: 0 16px 28px rgba(122, 109, 100, 0.12);"
            orb_inner = f'<img src="data:image/png;base64,{image_data}" alt="{label_name}" style="width:100%; height:100%; object-fit:cover; display:block; border-radius:999px;" />'
        item_markup += f"""
        <div class="fragrance-item">
            <div class="fragrance-orb" style="{orb_style}">
                {orb_inner}
            </div>
            <div class="fragrance-name" style="color:{theme['accent']};">{label_name}</div>
            <div class="fragrance-desc">{label_desc}</div>
        </div>
        """

    label_names = []
    for label in recommended_labels[:3]:
        label_name, _ = SCENT_LABEL_GUIDE.get(label, (label, ""))
        label_names.append(label_name)
    summary_labels = ", ".join(label_names)
    featured_product = products[0]
    featured_product_image_markup = build_product_image_markup(featured_product["name"], "featured-product-image")

    st.markdown(
        f"""
        <div class="fragrance-card" style="border-color:{hex_to_rgba(theme['accent'], 0.32)};">
            <div class="fragrance-title" style="color:{theme['accent']};">나에게 어울리는 향</div>
            <div class="fragrance-grid">{item_markup}</div>
            <div class="fragrance-summary">
                {mood['image_line']}와 {skin_profile}에 어울리는 향으로 {summary_labels}를 추천해요.
            </div>
            <div class="featured-product-card" style="border-color:{hex_to_rgba(theme['accent'], 0.28)};">
                {featured_product_image_markup}
                <div style="font-size:0.95rem; color:#6f645b; margin-bottom:0.45rem;">피부 특성을 고려한 제품 추천 ▼</div>
                <div class="featured-product-note">
                    {skin_profile}
                </div>
                <div class="featured-product-note" style="margin-top:0.35rem;">
                    {skin_recommendation_line}
                </div>
                <div style="font-size:1.45rem; font-weight:800; color:#2f2622; margin:1rem 0 0.55rem 0;">{featured_product['name']}</div>
                <div class="featured-product-note">{featured_product['notes']}</div>
                <div style="margin-top:1rem;">
                    <span class="season-cta" style="background:{theme['button']}; min-width:220px;">구매하러가기 →</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scent_loading_page() -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .stButton { display:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1rem; line-height:1.9; color:#675b54;">
                데이터 기반으로 당신에게 어울리는 향을 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.scent_analysis_pending:
        time.sleep(1.1)
        st.session_state.scent_analysis_pending = False
    st.session_state.page = "scent_result"
    st.rerun()


def render_share_actions(personal_color: str, recommended_labels: List[str], featured_product: Dict[str, str]) -> None:
    label_names = [SCENT_LABEL_GUIDE.get(label, (label, ""))[0] for label in recommended_labels[:3]]
    share_text = (
        f"유쏘풀 향 추천 결과\n"
        f"퍼스널 컬러: {personal_color}\n"
        f"추천 향: {', '.join(label_names)}\n"
        f"추천 제품: {featured_product['name']}"
    )
    share_text_js = share_text.replace("\\", "\\\\").replace("`", "\\`")
    components.html(
        f"""
        <div style="text-align:center; margin:-0.3rem 0 1.2rem 0;">
            <button id="share-result-btn" style="
                min-width:240px;
                border:none;
                border-radius:999px;
                padding:0.95rem 1.4rem;
                background:#ce2f97;
                color:white;
                font-weight:800;
                font-size:1.02rem;
                cursor:pointer;
                box-shadow:0 14px 24px rgba(122,109,100,0.18);
            ">결과 공유하기 →</button>
            <div id="share-result-message" style="margin-top:0.7rem; color:#8b674d; font-size:0.92rem;"></div>
        </div>
        <script>
        const shareText = `{share_text_js}`;
        const shareUrl = window.location.href;
        const msg = document.getElementById("share-result-message");
        document.getElementById("share-result-btn").onclick = async () => {{
            try {{
                if (navigator.share) {{
                    await navigator.share({{ title: "유쏘풀 향 추천 결과", text: shareText, url: shareUrl }});
                    msg.textContent = "공유 창을 열었어요.";
                }} else {{
                    await navigator.clipboard.writeText(shareText + "\\n" + shareUrl);
                    msg.textContent = "공유 기능을 지원하지 않아 문구와 링크를 복사했어요.";
                }}
            }} catch (error) {{
                msg.textContent = "공유가 취소되었거나 브라우저에서 제한되었어요.";
            }}
        }};
        </script>
        """,
        height=84,
    )


def build_skin_profile_phrase() -> str:
    skin_type = st.session_state.skin_type
    moisture = st.session_state.moisture_level
    temperature = st.session_state.temperature_level
    profile_key = (skin_type, moisture, temperature)
    return SKIN_PROFILE_LABELS.get(profile_key, "현재 피부 타입")


def build_skin_profile_recommendation_line() -> str:
    skin_type = st.session_state.skin_type
    moisture = st.session_state.moisture_level
    temperature = st.session_state.temperature_level
    profile_key = (skin_type, moisture, temperature)
    tone = SKIN_PROFILE_TONE.get(profile_key)
    if tone:
        return f"{tone} 당신의 피부 타입에 지속력이 좋은 제품을 추천해드릴게요."
    return "현재 피부 상태에 맞춰 지속력이 좋은 제품을 추천해드릴게요."


def render_result_page() -> None:
    render_step_progress()

    run_recommendation()
    selected_color = st.session_state.personal_color

    render_signature_palette_card(selected_color)
    recommended_products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    if recommended_products:
        render_result_featured_product_card(selected_color, recommended_products[0])
    center_left, center_mid, center_right = st.columns([1, 1.15, 1])
    with center_mid:
        if st.button("나에게 어울리는 향 찾기 →", key="to_scent_result", type="primary", use_container_width=True):
            st.session_state.scent_analysis_pending = True
            st.session_state.page = "scent_loading"
            st.rerun()

    left_btn, right_btn = st.columns(2)
    with left_btn:
        if st.button("← 이전", key="result_prev", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart", type="primary", use_container_width=True):
            reset_all()
            st.rerun()
    return


def render_scent_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color
    profile = PERSONAL_COLOR_PROFILES[selected_color]
    products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    recommended_labels = st.session_state.recommended_labels

    render_fragrance_section(selected_color, recommended_labels, products)
    render_share_actions(selected_color, recommended_labels, products[0])

    st.markdown("### 추천 제품")
    for product in products:
        product_image_markup = build_product_image_markup(product["name"])
        st.markdown(
            f"""
            <div class="product-card">
                <div style="display:flex; align-items:center; gap:0.9rem;">
                    {product_image_markup or f'<span class="icon-badge" style="width:48px; height:48px; font-size:1.4rem;">{profile["emoji"]}</span>'}
                    <div>
                        <div style="font-size:1.45rem; font-weight:800; color:#2f2622;">{product['name']}</div>
                        <div style="font-size:0.98rem; color:#8a776a; margin-top:0.35rem;">{product['match']}</div>
                    </div>
                </div>
                <div class="product-chip" style="margin-top:1rem;">{product['notes']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_btn, center_btn, right_btn = st.columns(3)
    with left_btn:
        if st.button("← 퍼컬 결과", key="back_to_result", use_container_width=True):
            go_to("result")
    with center_btn:
        if st.button("사진 다시 선택", key="back_to_face", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart_scent", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


def main() -> None:
    init_session()
    if st.session_state.analysis_pending and st.session_state.page == "face":
        st.session_state.page = "loading"
    if st.session_state.scent_analysis_pending and st.session_state.page == "result":
        st.session_state.page = "scent_loading"
    apply_global_style()
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "skin":
        render_choice_page(
            "피부 상태 체크",
            "지금 내 피부에\n가장 가까운 타입을 골라주세요",
            "평소 느끼는 피부 상태와 가장 가까운 항목 하나를 선택해 주세요.\n이 선택이 이후 향 추천의 기준이 됩니다.",
            SKIN_OPTIONS,
            "skin_type",
            "home",
            "moisture",
        )
    elif st.session_state.page == "moisture":
        render_choice_page(
            "수분감 체크",
            "평소 피부에 남는\n수분감을 골라주세요",
            "피부가 촉촉한 편인지, 건조한 편인지 골라주세요.\n향이 피부에 머무는 느낌과 지속감을 맞출 때 함께 참고해요.",
            MOISTURE_OPTIONS,
            "moisture_level",
            "skin",
            "temperature",
        )
    elif st.session_state.page == "temperature":
        render_choice_page(
            "피부 온도 체크",
            "피부 열감과 온도감도\n함께 골라주세요",
            "열감이 자주 도는 편인지, 차분하고 서늘한 편인지 선택해 주세요.\n발향 인상과 잔향 무드를 조정할 때 반영됩니다.",
            TEMPERATURE_OPTIONS,
            "temperature_level",
            "moisture",
            "face",
        )
    elif st.session_state.page == "face":
        render_face_page()
    elif st.session_state.page == "loading":
        render_loading_page()
    elif st.session_state.page == "result":
        render_result_page()
    elif st.session_state.page == "scent_loading":
        render_scent_loading_page()
    elif st.session_state.page == "scent_result":
        render_scent_result_page()

 
def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()

    render_step_progress()
    render_hero(
        "얼굴 사진 분석",
        "얼굴 사진으로\n퍼스널 컬러를 분석할게요.",
        "사진을 올리면 배경을 정리한 뒤 퍼스널 컬러를 분석해요.\n분석이 끝나면 결과 카드와 향 추천 화면으로 이어집니다.",
    )
    st.markdown("")
    _, center_area, _ = st.columns([0.9, 1.6, 0.9])
    with center_area:
        tab1, tab2 = st.tabs(["사진 업로드", "카메라로 찍기"])
        with tab1:
            uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        with tab2:
            camera_photo = st.camera_input("카메라 촬영")
            if camera_photo is not None:
                st.session_state.uploaded_image = camera_photo
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None

        if st.session_state.uploaded_image is not None:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="업로드된 사진", width=420)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="upload-status">사진 업로드가 완료됐어요. 결과 보기를 누르면 분석을 시작할게요.</div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        if st.button("← 이전", key="face_prev", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def render_signature_palette_card(personal_color: str) -> None:
    profile = PERSONAL_COLOR_PROFILES.get(personal_color)
    palette = PERSONAL_COLOR_PALETTES.get(personal_color)
    palette_image_path = PERSONAL_COLOR_PALETTE_IMAGES.get(personal_color)
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_path = PERSONAL_COLOR_REPRESENTATIVE_IMAGES.get(personal_color)

    if not profile or not palette:
        return

    image_data = image_to_base64(image_path) if image_path else None
    image_markup = ""
    if image_data:
        image_markup = (
            f'<img class="season-avatar" src="data:image/png;base64,{image_data}" alt="{personal_color} 대표 이미지" />'
        )

    keyword_pills = "".join(
        f'<span class="season-pill" style="background:{theme["accent"]};">{keyword}</span>'
        for keyword in profile.get("keywords", [])
    )
    description_lines = "<br>".join(profile.get("description_lines", []))
    colors = palette.get("colors", [])
    color_names = [color.get("name", "").strip() for color in colors if color.get("name", "").strip()]
    if not color_names:
        color_names = [keyword.replace("#", "") for keyword in profile.get("keywords", []) if keyword]
    palette_caption = " / ".join(color_names[:4])

    swatch_markup = "".join(
        f"""
        <div class="season-swatch-item">
            <div class="season-swatch-circle" style="background:{color['hex']};"></div>
            <div class="season-swatch-name">{color['name']}</div>
        </div>
        """
        for color in colors
    )

    if colors:
        palette_markup = f'<div class="season-swatch-grid">{swatch_markup}</div>'
    else:
        palette_markup = ""
        if palette_image_path and palette_image_path.exists():
            palette_image_data = image_to_base64(palette_image_path)
            if palette_image_data:
                palette_markup = (
                    f'<div style="max-width:420px; margin:0 auto 1.35rem auto;">'
                    f'<img src="data:image/png;base64,{palette_image_data}" alt="{personal_color} 팔레트" '
                    f'style="width:100%; display:block; border-radius:28px;" />'
                    f'<div class="season-swatch-name" style="margin-top:0.7rem; text-align:center;">{palette_caption}</div>'
                    f"</div>"
                )

    st.markdown(
        f"""
        <div class="season-card" style="
            border:1.6px solid {hex_to_rgba(theme['accent'], 0.72)};
            box-shadow: 0 24px 55px {hex_to_rgba(theme['accent'], 0.16)};
        ">
            {image_markup}
            <div class="season-title" style="color:{theme['accent']};">{personal_color}</div>
            <div class="season-pill-row">{keyword_pills}</div>
            <div class="season-body">{description_lines}</div>
            <div class="season-divider" style="color:{theme['accent']};"></div>
            <div class="season-subtitle" style="color:{theme['accent']};">어울리는 색상</div>
            {palette_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_featured_product_card(personal_color: str, featured_product: Dict[str, str]) -> None:
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_markup = build_product_image_markup(featured_product["name"], "featured-product-image")
    skin_profile = build_skin_profile_phrase()
    skin_recommendation_line = build_skin_profile_recommendation_line()
    body_markup = "".join(
        part
        for part in [
            image_markup,
            f'<div style="font-size:1.42rem; font-weight:800; color:#2f2622; margin:0.2rem 0 0.55rem 0;">{featured_product["name"]}</div>',
            f'<div class="featured-product-note">{featured_product["notes"]}</div>',
            f'<div class="featured-product-note" style="margin-top:0.65rem;">{skin_profile}</div>',
            f'<div class="featured-product-note" style="margin-top:0.35rem;">{skin_recommendation_line}</div>',
        ]
        if part
    )
    st.markdown(
        f"""
        <div class="result-featured-product-card" style="border-color:{hex_to_rgba(theme['accent'], 0.24)};">
            <div style="font-size:0.92rem; font-weight:700; color:{theme['accent']}; margin-bottom:0.55rem;">유쏘풀 추천 향수</div>
            {body_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fragrance_section(personal_color: str, recommended_labels: List[str], products: List[Dict[str, str]]) -> None:
    mood = PERSONAL_COLOR_MOOD_GUIDE.get(personal_color)
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    skin_profile = build_skin_profile_phrase()
    if not recommended_labels or not mood:
        return

    item_markup = ""
    for label in recommended_labels[:3]:
        _, label_desc = SCENT_LABEL_GUIDE.get(label, (label, "설명이 아직 준비되지 않았어요."))
        label_name = label
        image_path = get_scent_label_image_path(label)
        image_data = image_to_base64(image_path) if image_path else None
        orb_inner = label_name
        orb_style = f"background: radial-gradient(circle at 30% 25%, {hex_to_rgba(theme['accent'], 0.78)}, {theme['button']});"
        if image_data:
            orb_style = "background: transparent; box-shadow: 0 16px 28px rgba(122, 109, 100, 0.12);"
            orb_inner = f'<img src="data:image/png;base64,{image_data}" alt="{label_name}" style="width:100%; height:100%; object-fit:cover; display:block; border-radius:999px;" />'
        item_markup += f"""
        <div class="fragrance-item">
            <div class="fragrance-orb" style="{orb_style}">
                {orb_inner}
            </div>
            <div class="fragrance-name" style="color:{theme['accent']};">{label_name}</div>
            <div class="fragrance-desc">{label_desc}</div>
        </div>
        """

    label_names = []
    for label in recommended_labels[:3]:
        label_names.append(label)
    summary_labels = ", ".join(label_names)

    st.markdown(
        f"""
        <div class="fragrance-card" style="border-color:{hex_to_rgba(theme['accent'], 0.32)};">
            <div class="fragrance-title" style="color:{theme['accent']};">나에게 어울리는 향</div>
            <div class="fragrance-grid">{item_markup}</div>
            <div class="fragrance-summary">
                {mood['image_line']}와 {skin_profile}에 어울리는 향으로 {summary_labels}를 추천해요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_native_featured_product_card(personal_color: str, featured_product: Dict[str, str]) -> None:
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_markup = build_product_image_markup(featured_product["name"], "featured-product-image")
    skin_profile = build_skin_profile_phrase()
    skin_recommendation_line = build_skin_profile_recommendation_line()
    text_parts = [
        f'<div class="featured-product-note" style="margin-top:0.65rem;">{skin_profile}</div>',
        f'<div class="featured-product-note" style="margin-top:0.35rem;">{skin_recommendation_line}</div>',
    ]
    if not image_markup:
        text_parts = [
            f'<div style="font-size:1.42rem; font-weight:800; color:#2f2622; margin:0.95rem 0 0.55rem 0;">{featured_product["name"]}</div>',
            f'<div class="featured-product-note">{featured_product["notes"]}</div>',
            *text_parts,
        ]
    featured_body = "".join([part for part in [image_markup, *text_parts] if part])
    st.markdown(
        f"""
        <div class="result-featured-product-card" style="border-color:{hex_to_rgba(theme['accent'], 0.24)};">
            <div style="font-size:0.92rem; font-weight:700; color:{theme['accent']}; margin-bottom:0.55rem;">유쏘풀 추천 향수</div>
            {featured_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pick_featured_product(products: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for product in products:
        if get_product_image_path(product["name"]):
            return product
    return products[0] if products else None


def render_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color

    render_signature_palette_card(selected_color)

    _, center_mid, _ = st.columns([1, 1.15, 1])
    with center_mid:
        if st.button("나에게 어울리는 향 찾기 →", key="to_scent_result", type="primary", use_container_width=True):
            st.session_state.scent_analysis_pending = True
            st.session_state.page = "scent_loading"
            st.rerun()

    left_btn, right_btn = st.columns(2)
    with left_btn:
        if st.button("← 이전", key="result_prev", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


def render_scent_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color
    products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    recommended_labels = st.session_state.recommended_labels

    render_fragrance_section(selected_color, recommended_labels, products)
    featured_product = pick_featured_product(products)
    if featured_product:
        render_native_featured_product_card(selected_color, featured_product)
        render_share_actions(selected_color, recommended_labels, featured_product)

    left_btn, center_btn, right_btn = st.columns(3)
    with left_btn:
        if st.button("← 퍼컬 결과", key="back_to_result", use_container_width=True):
            go_to("result")
    with center_btn:
        if st.button("사진 다시 선택", key="back_to_face", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart_scent", type="primary", use_container_width=True):
            reset_all()
            st.rerun()

FEEDBACK_DIR = APP_DIR / "feedback_data"
CONSENTED_IMAGE_DIR = FEEDBACK_DIR / "consented_images"
FEEDBACK_LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"
FEEDBACK_SHEET_WORKSHEET = "feedback"


def ensure_feedback_dirs() -> None:
    FEEDBACK_DIR.mkdir(exist_ok=True)
    CONSENTED_IMAGE_DIR.mkdir(exist_ok=True)


def get_feedback_sheet_settings() -> Optional[Dict[str, object]]:
    try:
        feedback_storage = st.secrets.get("feedback_storage", {})
    except Exception:
        feedback_storage = {}

    if not feedback_storage:
        return None

    provider = str(feedback_storage.get("provider", "")).strip().lower()
    sheet_id = str(feedback_storage.get("sheet_id", "")).strip()
    worksheet = str(feedback_storage.get("worksheet", FEEDBACK_SHEET_WORKSHEET)).strip() or FEEDBACK_SHEET_WORKSHEET

    if provider != "google_sheets" or not sheet_id:
        return None

    try:
        service_account = st.secrets.get("google_service_account") or st.secrets.get("gcp_service_account")
    except Exception:
        service_account = None

    if not service_account:
        return None

    return {
        "provider": provider,
        "sheet_id": sheet_id,
        "worksheet": worksheet,
        "service_account": dict(service_account),
    }


def append_feedback_to_google_sheet(row: Dict[str, str]) -> str:
    settings = get_feedback_sheet_settings()
    if not settings:
        return ""

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return ""

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(settings["service_account"], scopes=scopes)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(str(settings["sheet_id"]))

    try:
        worksheet = spreadsheet.worksheet(str(settings["worksheet"]))
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=str(settings["worksheet"]), rows=1000, cols=20)

    headers = [
        "timestamp",
        "uploaded_name",
        "predicted_label",
        "predicted_confidence",
        "user_label",
        "is_match",
        "consent",
        "feedback_comment",
        "saved_image_path",
    ]

    existing_header = worksheet.row_values(1)
    if not existing_header:
        worksheet.append_row(headers)
    elif existing_header != headers:
        worksheet.clear()
        worksheet.append_row(headers)

    worksheet.append_row([row.get(header, "") for header in headers])
    return f"Google Sheets / {settings['worksheet']}"


def ensure_integrated_feedback_state() -> None:
    uploaded_name = getattr(st.session_state.uploaded_image, "name", "camera_capture")
    context_key = f"{uploaded_name}:{st.session_state.personal_color}"
    defaults = {
        "feedback_context_key": context_key,
        "feedback_match_answer": "맞아요",
        "feedback_actual_label": st.session_state.personal_color,
        "feedback_consent_answer": "비동의",
        "feedback_saved": False,
        "feedback_saved_message": None,
    }
    if st.session_state.get("feedback_context_key") != context_key:
        for key, value in defaults.items():
            st.session_state[key] = value
        return
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_integrated_feedback_log(
    *,
    uploaded_name: str,
    predicted_label: str,
    predicted_confidence: Optional[float],
    user_label: str,
    is_match: bool,
    consent: bool,
    saved_image_path: str = "",
) -> None:
    ensure_feedback_dirs()
    is_new = not FEEDBACK_LOG_PATH.exists()
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "uploaded_name",
                "predicted_label",
                "predicted_confidence",
                "user_label",
                "is_match",
                "consent",
                "saved_image_path",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "uploaded_name": uploaded_name,
                "predicted_label": predicted_label,
                "predicted_confidence": f"{predicted_confidence:.4f}" if predicted_confidence is not None else "",
                "user_label": user_label,
                "is_match": "yes" if is_match else "no",
                "consent": "yes" if consent else "no",
                "saved_image_path": saved_image_path,
            }
        )


def save_integrated_feedback_image(uploaded_file, user_label: str) -> str:
    ensure_feedback_dirs()
    original_suffix = Path(getattr(uploaded_file, "name", "")).suffix or ".png"
    safe_label = normalize_asset_key(user_label) or "unknown"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}_{time.time_ns() % 1000000000}{original_suffix}"
    target_path = CONSENTED_IMAGE_DIR / filename
    target_path.write_bytes(uploaded_file.getvalue())
    return str(target_path)


def submit_integrated_feedback(actual_label: str, is_match: bool, consent: bool) -> None:
    uploaded_file = st.session_state.uploaded_image
    if uploaded_file is None:
        return

    saved_image_path = ""
    if consent:
        saved_image_path = save_integrated_feedback_image(uploaded_file, actual_label)

    save_integrated_feedback_log(
        uploaded_name=getattr(uploaded_file, "name", "camera_capture"),
        predicted_label=st.session_state.personal_color,
        predicted_confidence=st.session_state.prediction_confidence,
        user_label=actual_label,
        is_match=is_match,
        consent=consent,
        saved_image_path=saved_image_path,
    )

    st.session_state.feedback_saved = True
    if consent:
        st.session_state.feedback_saved_message = (
            "피드백과 이미지 사용 동의가 저장됐어요.\n"
            f"로그 저장 위치: {FEEDBACK_LOG_PATH}\n"
            f"이미지 저장 위치: {saved_image_path}"
        )
    else:
        st.session_state.feedback_saved_message = (
            "피드백은 저장했고, 이미지는 학습 데이터로 사용하지 않도록 기록해둘게요.\n"
            f"로그 저장 위치: {FEEDBACK_LOG_PATH}"
        )


def render_integrated_feedback_section() -> None:
    ensure_integrated_feedback_state()
    predicted_label = st.session_state.personal_color
    confidence = st.session_state.prediction_confidence
    class_options = list(load_personal_color_assets()["classes"])

    st.markdown(
        """
        <div style="
            max-width:760px;
            margin:1.5rem auto 1rem auto;
            background:#12151c;
            color:#f7f4ef;
            border:1px solid rgba(255,255,255,0.08);
            border-radius:28px;
            padding:1.6rem 1.5rem 1.35rem 1.5rem;
            box-shadow:0 22px 38px rgba(0,0,0,0.18);
        ">
            <div style="font-size:1.55rem; font-weight:900; margin-bottom:0.45rem;">결과 확인</div>
            <div style="font-size:0.98rem; color:#cbc3b8; line-height:1.7;">
                퍼스널 컬러와 향 추천이 끝났다면 마지막으로 결과가 맞는지 알려주세요.
                맞는 결과는 정답 데이터 후보로, 다른 결과는 수정 라벨과 함께 학습 검증용 데이터 후보로 남길 수 있어요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"1. 예측 결과가 **{predicted_label}** 인데 맞나요?")
    if confidence is not None:
        st.caption(f"예측 신뢰도: {confidence:.1%}")

    st.radio(
        "예측 결과 확인",
        ["맞아요", "아니에요"],
        horizontal=True,
        key="feedback_match_answer",
        label_visibility="collapsed",
    )

    is_match = st.session_state.feedback_match_answer == "맞아요"
    actual_label = predicted_label

    if not is_match:
        st.markdown("2. 아니라면 실제 퍼스널 컬러가 무엇인지 알려주세요.")
        actual_label = st.selectbox(
            "실제 퍼스널 컬러",
            class_options,
            index=class_options.index(predicted_label) if predicted_label in class_options else 0,
            key="feedback_actual_label",
        )

    consent_question = (
        "3. 결과가 맞으니 이 이미지와 라벨을 정답 데이터로 학습에 사용해도 될까요?"
        if is_match
        else "3. 결과는 달랐지만, 수정한 라벨과 함께 이 이미지를 모델 학습과 검증에 사용해도 될까요?"
    )
    st.markdown(consent_question)

    st.radio(
        "학습 데이터 사용 동의",
        ["동의", "비동의"],
        horizontal=True,
        key="feedback_consent_answer",
        label_visibility="collapsed",
    )

    submit_col, reset_col = st.columns(2)
    with submit_col:
        if st.button("피드백 저장", type="primary", use_container_width=True, key="integrated_feedback_submit"):
            submit_integrated_feedback(
                actual_label=actual_label,
                is_match=is_match,
                consent=st.session_state.feedback_consent_answer == "동의",
            )
            st.rerun()
    with reset_col:
        if st.button("응답 다시 선택", use_container_width=True, key="integrated_feedback_reset"):
            st.session_state.feedback_match_answer = "맞아요"
            st.session_state.feedback_actual_label = predicted_label
            st.session_state.feedback_consent_answer = "비동의"
            st.session_state.feedback_saved = False
            st.session_state.feedback_saved_message = None
            st.rerun()

    if st.session_state.feedback_saved and st.session_state.feedback_saved_message:
        st.success(st.session_state.feedback_saved_message)


def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return
    st.markdown("<div style='height:2.1rem;'></div>", unsafe_allow_html=True)
    current = current_step_index()
    cols = st.columns(len(STEP_PAGES))
    for idx, (col, page) in enumerate(zip(cols, STEP_PAGES), start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="
                        width:50px;
                        height:50px;
                        margin:0 auto 0.55rem auto;
                        border-radius:999px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:{bubble_bg};
                        color:white;
                        font-weight:800;
                        font-size:1.25rem;
                    ">{bubble_text}</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{label_color};">{STEP_LABELS[page]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:3.1rem;'></div>", unsafe_allow_html=True)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-page-wrap {
            max-width: 780px;
            margin: 0 auto;
        }
        .choice-page-wrap .hero-card {
            text-align: center;
            padding: 2.25rem 2.25rem 2rem 2.25rem;
        }
        .choice-page-wrap .hero-card > div:first-child,
        .choice-page-wrap .hero-card > div:nth-child(2),
        .choice-page-wrap .hero-card > div:nth-child(3) {
            text-align: center;
        }
        .choice-layout-card {
            background: rgba(255, 252, 248, 0.96);
            border: 1.35px solid #e2cdbc;
            border-radius: 22px;
            padding: 1rem 1.1rem 0.92rem 1.1rem;
            box-shadow: 0 12px 26px rgba(199, 171, 145, 0.08);
            margin-bottom: 0.55rem;
        }
        .choice-layout-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #2f2622;
            margin-bottom: 0.4rem;
        }
        .choice-layout-copy {
            color: #7d7068;
            font-size: 0.98rem;
            line-height: 1.7;
        }
        .choice-page-wrap div.stButton > button {
            min-height: 44px !important;
            font-size: 0.98rem !important;
        }
        .choice-action-row {
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="choice-page-wrap">', unsafe_allow_html=True)
    render_hero(kicker, title, description)
    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        st.markdown(
            f"""
            <div class="choice-layout-card" style="border-color:{'#8c63ff' if is_active else '#e2cdbc'};">
                <div class="choice-layout-title">{item['title']}</div>
                <div class="choice-layout-copy">{item['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="choice-action-row">', unsafe_allow_html=True)
        if st.button(
            item["button"],
            key=f"{state_key}_layout_pick_{index}",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        if st.button("← 이전", key=f"prev_{state_key}_layout", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음 단계로 →",
            key=f"next_{state_key}_layout",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    render_hero(kicker, title, description)
    st.markdown(
        """
        <style>
        .choice-shell {
            max-width: 760px;
            margin: 0 auto;
        }
        .choice-shell div.stButton > button {
            min-height: 36px !important;
            font-size: 0.96rem !important;
        }
        .choice-hint {
            margin: 0.28rem 0 0.85rem 0.35rem;
            color: #8a776a;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown('<div class="choice-shell">', unsafe_allow_html=True)
    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        content = get_choice_display_content(state_key, index)
        is_active = selected == item["key"]
        if st.button(
            content["title"] or item["title"],
            key=f"{state_key}_choice_compact_latest_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown(
            f'<div class="choice-hint">{content["hint"] or item["description"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_compact_latest", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_compact_latest",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_integrated_feedback_section() -> None:
    ensure_integrated_feedback_state()
    predicted_label = st.session_state.personal_color
    class_options = list(load_personal_color_assets()["classes"])
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stTextArea"] label p {
            color:#8a776a !important;
            font-weight:700 !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            background: rgba(255,251,247,0.96) !important;
            border: 1.4px solid #dcc7b6 !important;
            border-radius: 20px !important;
            color:#2f2622 !important;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 132px !important;
            box-shadow: 0 14px 28px rgba(199,171,145,0.10) !important;
            padding: 1rem 1.05rem !important;
            line-height: 1.7 !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color:#8b674d !important;
            opacity:1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    body_left, body_center, body_right = st.columns([0.45, 5.2, 0.45])
    with body_center:
        st.markdown("##### 예측 결과가 맞나요?")
        st.markdown(f"**{predicted_label}**")
        render_feedback_choice_buttons("feedback_match_answer", ["맞아요", "아니에요"], "feedback_match_latest3")
        is_match = st.session_state.feedback_match_answer == "맞아요"
        actual_label = predicted_label
        if not is_match:
            st.markdown("##### 실제 퍼스널 컬러를 알려주세요.")
            actual_label = st.selectbox(
                "실제 퍼스널 컬러",
                class_options,
                index=class_options.index(predicted_label) if predicted_label in class_options else 0,
                key="feedback_actual_label_latest3",
            )
            st.session_state.feedback_actual_label = actual_label
        st.markdown(
            "##### 이 이미지를 학습 데이터로 사용해도 될까요?"
            if is_match
            else "##### 수정한 라벨과 함께 이 이미지를 학습과 검증에 사용해도 될까요?"
        )
        render_feedback_choice_buttons("feedback_consent_answer", ["동의", "비동의"], "feedback_consent_latest3")
        st.markdown("##### 추가적인 의견을 적어주세요. 피드백은 소중한 의견이 됩니다.")
        st.text_area(
            "추가 피드백",
            key="feedback_comment",
            placeholder="추가적인 의견을 자유롭게 적어주세요. 남겨주신 피드백은 더 나은 결과를 만드는 데 도움이 됩니다.",
            height=132,
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        _, submit_col, _ = st.columns([1, 1.6, 1])
        with submit_col:
            if st.button("피드백 저장", type="primary", use_container_width=True, key="integrated_feedback_submit_latest3"):
                submit_integrated_feedback(
                    actual_label=actual_label,
                    is_match=is_match,
                    consent=st.session_state.feedback_consent_answer == "동의",
                )
                st.rerun()
    st.markdown("<div style='height:4rem;'></div>", unsafe_allow_html=True)
    if st.session_state.feedback_saved and st.session_state.feedback_saved_message:
        st.success(st.session_state.feedback_saved_message)


def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return
    current = current_step_index()
    cols = st.columns(len(STEP_PAGES))
    for idx, (col, page) in enumerate(zip(cols, STEP_PAGES), start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="
                        width:50px;
                        height:50px;
                        margin:0 auto 0.55rem auto;
                        border-radius:999px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:{bubble_bg};
                        color:white;
                        font-weight:800;
                        font-size:1.25rem;
                    ">{bubble_text}</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{label_color};">{STEP_LABELS[page]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)


def render_loading_page() -> None:
    if st.session_state.uploaded_image is None:
        go_to("face")
        return

    st.markdown(
        """
        <style>
        .stButton,
        .stTabs,
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"],
        [data-testid="stImage"],
        .upload-preview,
        .upload-status {
            display:none !important;
        }
        .block-container {
            padding-top: 6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel" style="margin-top:0.2rem;">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1rem; line-height:1.9; color:#675b54;">
                업로드한 사진을 바탕으로 퍼스널 컬러를 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.analysis_pending:
        time.sleep(1.0)
        run_recommendation()
        st.session_state.analysis_pending = False
    st.session_state.page = "result"
    st.rerun()


def render_scent_loading_page() -> None:
    st.markdown(
        """
        <style>
        .stButton { display:none !important; }
        .block-container {
            padding-top: 6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel" style="margin-top:0.2rem;">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1rem; line-height:1.9; color:#675b54;">
                데이터 기반으로 나에게 어울리는 향을 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.scent_analysis_pending:
        time.sleep(1.1)
        st.session_state.scent_analysis_pending = False
    st.session_state.page = "scent_result"
    st.rerun()


def get_choice_display_content(state_key: str, index: int) -> Dict[str, str]:
    content_map = {
        "skin_type": [
            {"title": "지성", "hint": "유분이 비교적 많이 느껴지는 피부예요."},
            {"title": "건성", "hint": "건조함과 당김이 자주 느껴지는 피부예요."},
            {"title": "민감성", "hint": "자극과 붉어짐에 예민한 피부예요."},
            {"title": "잘 모르겠어요", "hint": "아직 피부 타입을 딱 정하기 어려운 편이에요."},
        ],
        "moisture_level": [
            {"title": "건조해요", "hint": "수분이 쉽게 날아가고 메마름이 느껴져요."},
            {"title": "건조하지 않아요", "hint": "수분감이 비교적 유지되는 편이에요."},
        ],
        "temperature_level": [
            {"title": "열감이 높은 편", "hint": "피부에 열감이 오래 남는 편이에요."},
            {"title": "열감이 낮은 편", "hint": "차분하고 서늘하게 느껴지는 편이에요."},
        ],
    }
    options = content_map.get(state_key, [])
    if 0 <= index < len(options):
        return options[index]
    return {"title": "", "hint": ""}


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    render_hero(kicker, title, description)
    st.markdown(
        """
        <style>
        .choice-shell {
            max-width: 760px;
            margin: 0 auto;
        }
        .choice-hint {
            margin: 0.35rem 0 1.15rem 0.35rem;
            color: #8a776a;
            font-size: 0.94rem;
            line-height: 1.65;
        }
        .choice-shell div.stButton > button {
            min-height: 68px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown('<div class="choice-shell">', unsafe_allow_html=True)
    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        content = get_choice_display_content(state_key, index)
        is_active = selected == item["key"]
        if st.button(
            content["title"] or item["title"],
            key=f"{state_key}_choice_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown(
            f'<div class="choice-hint">{content["hint"] or item["description"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()

    render_step_progress()
    render_hero(
        "얼굴 사진 분석",
        "얼굴 사진으로\n퍼스널 컬러를 분석할게요.",
        "사진을 올리면 배경을 정리한 뒤 퍼스널 컬러를 분석해요.\n분석이 끝나면 결과 카드와 향 추천 화면으로 이어집니다.",
    )
    st.markdown(
        """
        <style>
        .face-upload-shell {
            max-width: 620px;
            margin: 0 auto;
        }
        .face-upload-shell div[data-testid="stTabs"] {
            width: 100%;
        }
        .face-upload-shell .stTabs [data-baseweb="tab-list"] {
            justify-content: flex-end;
            margin-bottom: 0.9rem;
        }
        .face-upload-shell [data-testid="stFileUploader"],
        .face-upload-shell [data-testid="stCameraInput"] {
            width: 100%;
            margin-top: 0.6rem;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 251, 247, 0.96) !important;
            border: 1.4px dashed #d8bca5 !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.10) !important;
            min-height: 160px !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] section {
            background: transparent !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] * {
            color: #6f645b !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] small,
        .face-upload-shell [data-testid="stFileUploaderDropzone"] span,
        .face-upload-shell [data-testid="stFileUploaderDropzone"] p {
            color: #6f645b !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] svg {
            fill: #b48767 !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button {
            background: #a97757 !important;
            color: #ffffff !important;
            border: 1px solid #a97757 !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button:hover {
            background: #956848 !important;
            border-color: #956848 !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
        }
        .face-upload-shell [data-testid="stCameraInput"] > div {
            background: rgba(255, 251, 247, 0.96) !important;
            border: 1.4px dashed #d8bca5 !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.10) !important;
            padding: 0.85rem !important;
        }
        .face-upload-shell [data-testid="stCameraInput"] button {
            background: #fffaf6 !important;
            color: #9f6c45 !important;
            border: 1px solid #d8bca5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    _, center_area, _ = st.columns([1.1, 2.8, 1.1])
    with center_area:
        st.markdown('<div class="face-upload-shell">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["사진 업로드", "카메라로 찍기"])
        with tab1:
            uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        with tab2:
            camera_photo = st.camera_input("카메라 촬영")
            if camera_photo is not None:
                st.session_state.uploaded_image = camera_photo
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None

        if st.session_state.uploaded_image is not None:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="업로드된 사진", width=520)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="upload-status">사진 업로드가 완료됐어요. 결과 보기를 누르면 분석이 시작됩니다.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        if st.button("이전", key="face_prev", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def render_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color

    render_signature_palette_card(selected_color)

    _, center_mid, _ = st.columns([1, 1.15, 1])
    with center_mid:
        if st.button("나에게 어울리는 향 찾기 →", key="to_scent_result", type="primary", use_container_width=True):
            st.session_state.scent_analysis_pending = True
            st.session_state.page = "scent_loading"
            st.rerun()

    left_btn, right_btn = st.columns(2)
    with left_btn:
        if st.button("이전", key="result_prev", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


def render_scent_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color
    products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    recommended_labels = st.session_state.recommended_labels

    render_fragrance_section(selected_color, recommended_labels, products)
    featured_product = pick_featured_product(products)
    if featured_product:
        render_native_featured_product_card(selected_color, featured_product)
        render_share_actions(selected_color, recommended_labels, featured_product)

    render_integrated_feedback_section()

    left_btn, center_btn, right_btn = st.columns(3)
    with left_btn:
        if st.button("이전 결과", key="back_to_result", use_container_width=True):
            go_to("result")
    with center_btn:
        if st.button("사진 다시 선택", key="back_to_face", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart_scent", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


def render_feedback_choice_buttons(state_key: str, options: List[str], key_prefix: str) -> None:
    columns = st.columns(len(options))
    for index, (column, option) in enumerate(zip(columns, options)):
        is_active = st.session_state.get(state_key) == option
        with column:
            if st.button(
                option,
                key=f"{key_prefix}_{state_key}_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state[state_key] = option
                st.rerun()


def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()

    render_step_progress()
    render_hero(
        "얼굴 사진 분석",
        "얼굴 사진으로\n퍼스널 컬러를 분석할게요.",
        "사진을 올리면 배경을 정리한 뒤 퍼스널 컬러를 분석해요.\n분석이 끝나면 결과 카드와 향 추천 화면으로 이어집니다.",
    )
    st.markdown(
        """
        <style>
        .face-upload-shell {
            max-width: 430px;
            margin: 0 auto;
        }
        .face-upload-shell div[data-testid="stTabs"] {
            width: 100%;
        }
        .face-upload-shell div[data-testid="stFileUploader"] {
            width: 100%;
        }
        .face-upload-shell div[data-testid="stCameraInput"] {
            width: 100%;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 251, 247, 0.96) !important;
            border: 1.4px dashed #d8bca5 !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.10) !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] section {
            background: transparent !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] * {
            color: #6f645b !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] small,
        .face-upload-shell [data-testid="stFileUploaderDropzone"] span,
        .face-upload-shell [data-testid="stFileUploaderDropzone"] p {
            color: #6f645b !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] svg {
            fill: #b48767 !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button {
            background: #fffaf6 !important;
            color: #ffffff !important;
            border: 1px solid #d8bca5 !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button:hover {
            background: #f7ede4 !important;
            border-color: #cfa889 !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
        }
        .face-upload-shell [data-testid="stCameraInput"] > div {
            background: rgba(255, 251, 247, 0.96) !important;
            border: 1.4px dashed #d8bca5 !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.10) !important;
            padding: 0.4rem !important;
        }
        .face-upload-shell [data-testid="stCameraInput"] button {
            background: #fffaf6 !important;
            color: #9f6c45 !important;
            border: 1px solid #d8bca5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    _, center_area, _ = st.columns([1.5, 2.1, 1.5])
    with center_area:
        st.markdown('<div class="face-upload-shell">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["사진 업로드", "카메라로 찍기"])
        with tab1:
            uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        with tab2:
            camera_photo = st.camera_input("카메라 촬영")
            if camera_photo is not None:
                st.session_state.uploaded_image = camera_photo
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None

        if st.session_state.uploaded_image is not None:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="업로드된 사진", width=420)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="upload-status">사진 업로드가 완료됐어요. 결과 보기를 누르면 분석이 시작됩니다.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        if st.button("이전", key="face_prev", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def toggle_choice_info(state_key: str, item_key: str) -> None:
    info_state_key = f"{state_key}_info_key"
    current = st.session_state.get(info_state_key)
    st.session_state[info_state_key] = None if current == item_key else item_key


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    render_hero(kicker, title, description)
    st.markdown(
        """
        <style>
        div[data-testid="column"] .skin-option-main button {
            border-top-right-radius: 0 !important;
            border-bottom-right-radius: 0 !important;
            border-right: none !important;
            min-height: 74px !important;
            background: #fffaf6 !important;
            color: #332823 !important;
            border: 1px solid #d8bca5 !important;
            box-shadow: none !important;
        }
        div[data-testid="column"] .skin-option-main button[kind="primary"] {
            background: #f8afa5 !important;
            color: #332823 !important;
            border-color: #f8afa5 !important;
        }
        div[data-testid="column"] .skin-option-info button {
            border-top-left-radius: 0 !important;
            border-bottom-left-radius: 0 !important;
            min-height: 74px !important;
            width: 100% !important;
            padding: 0 !important;
            font-size: 1.15rem !important;
            font-weight: 800 !important;
            background: #fffaf6 !important;
            color: #8f857d !important;
            border: 1px solid #d8bca5 !important;
            box-shadow: none !important;
            margin-left: -0.35rem !important;
        }
        div[data-testid="column"] .skin-option-info button:hover {
            background: #fff4ec !important;
            color: #7e7269 !important;
            border-color: #d8bca5 !important;
        }
        div[data-testid="column"] .skin-option-info button p {
            color: inherit !important;
            font-weight: 800 !important;
        }
        .skin-info-panel {
            margin: -0.15rem 0 0.8rem 0;
            padding: 1rem 1.15rem;
            background: rgba(255, 251, 247, 0.92);
            border: 1px solid #ead8c9;
            border-radius: 20px;
            color: #6f645b;
            line-height: 1.75;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    selected = st.session_state.get(state_key)
    info_state_key = f"{state_key}_info_key"
    opened_info_key = st.session_state.get(info_state_key)
    for index, item in enumerate(options):
        option_cols = st.columns([8.9, 1.1], gap="small")
        with option_cols[0]:
            is_active = selected == item["key"]
            st.markdown('<div class="skin-option-main">', unsafe_allow_html=True)
            if st.button(
                item["title"],
                key=f"{state_key}_choice_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with option_cols[1]:
            st.markdown('<div class="skin-option-info">', unsafe_allow_html=True)
            if st.button("i", key=f"{state_key}_info_{index}", use_container_width=True):
                toggle_choice_info(state_key, item["key"])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if opened_info_key == item["key"]:
            st.markdown(
                f"""
                <div class="skin-info-panel">
                    <div style="font-weight:800; color:#2f2622; margin-bottom:0.4rem;">{item['title']}</div>
                    <div>{item['description']}</div>
                    <div style="margin-top:0.45rem; color:#8a776a;">{item['examples']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_home() -> None:
    steps = [
        ("피부 상태 확인", "피부 타입, 수분감, 피부 온도를 차례로 체크해 현재 피부 컨디션을 파악해요."),
        ("퍼스널 컬러 분석", "얼굴 사진을 바탕으로 퍼스널 컬러를 분석하고 결과 카드를 보여드려요."),
        ("어울리는 향 추천", "퍼스널 컬러 결과와 피부 데이터를 함께 반영해 어울리는 향을 골라드려요."),
        ("향수 제품 추천", "추천된 어울리는 향 계열과 피부 특성을 바탕으로 유쏘풀 제품까지 이어서 추천해드려요."),
    ]
    flow_markup = ""
    for index, (title, desc) in enumerate(steps):
        flow_markup += (
            '<div class="home-flow-step">'
            f'<div class="home-flow-title">{title}</div>'
            f'<div class="home-flow-desc">{desc}</div>'
            "</div>"
        )
        if index < len(steps) - 1:
            flow_markup += (
                '<div class="home-flow-arrow-wrap">'
                '<div class="home-flow-line"></div>'
                '<div class="home-flow-arrow"></div>'
                "</div>"
            )

    home_html = (
        "<style>"
        ".home-page-offset{height:2.6rem;}"
        ".home-hero-wide{background:linear-gradient(180deg,rgba(255,252,248,0.96),rgba(255,248,241,0.94));border:1.6px solid #d8c0ad;border-radius:38px;padding:2.7rem 2.6rem 2.4rem 2.6rem;box-shadow:0 28px 60px rgba(182,148,121,0.16);margin-bottom:1.9rem;position:relative;overflow:hidden;}"
        ".home-hero-wide:before{content:'';position:absolute;right:-80px;top:-80px;width:240px;height:240px;border-radius:999px;background:radial-gradient(circle,rgba(201,162,127,0.22),rgba(201,162,127,0));}"
        ".home-hero-kicker{color:#9f6c45;font-weight:800;letter-spacing:.16em;font-size:.82rem;margin-bottom:1rem;position:relative;z-index:1;}"
        ".home-hero-title{font-size:2.6rem;line-height:1.18;font-weight:900;color:#251d19;margin-bottom:1rem;white-space:pre-line;letter-spacing:-.03em;position:relative;z-index:1;}"
        ".home-hero-desc{max-width:760px;font-size:1.02rem;line-height:1.9;color:#6d5b4f;position:relative;z-index:1;}"
        ".home-flow-head{margin:.2rem 0 1rem 0;font-size:1rem;font-weight:800;color:#8e6a52;letter-spacing:-.01em;}"
        ".home-flow-step{background:rgba(255,252,248,0.94);border:1.4px solid #dcc7b6;border-radius:24px;padding:1.2rem 1.25rem 1.05rem 1.25rem;box-shadow:0 14px 30px rgba(199,171,145,0.10);position:relative;overflow:hidden;}"
        ".home-flow-step:before{content:'';position:absolute;left:1.05rem;top:0;width:72px;height:5px;border-radius:0 0 999px 999px;background:linear-gradient(90deg,#b67c56,#d9b08c);}"
        ".home-flow-title{font-size:1.42rem;font-weight:800;color:#2b221e;letter-spacing:-.02em;padding-top:.3rem;}"
        ".home-flow-desc{margin-top:.42rem;font-size:.96rem;line-height:1.72;color:#6f645b;}"
        ".home-flow-arrow-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;margin:.12rem 0;}"
        ".home-flow-line{width:8px;height:16px;background:linear-gradient(180deg,#c7976f,#b57b55);border-radius:999px;}"
        ".home-flow-arrow{width:0;height:0;border-left:18px solid transparent;border-right:18px solid transparent;border-top:18px solid #b57b55;filter:drop-shadow(0 8px 12px rgba(181,123,85,0.18));}"
        "@media (max-width:768px){.home-page-offset{height:1.4rem;}.home-hero-wide{padding:2rem 1.4rem;}.home-hero-title{font-size:2rem;}.home-flow-title{font-size:1.18rem;}.home-flow-step{padding:1.05rem .95rem .9rem .95rem;border-radius:20px;}.home-flow-step:before{left:.9rem;width:56px;height:4px;}.home-flow-desc{font-size:.92rem;line-height:1.62;}}"
        "</style>"
        '<div class="home-page-offset"></div>'
        '<div class="home-hero-wide">'
        '<div class="home-hero-kicker">유쏘풀 향 추천</div>'
        '<div class="home-hero-title">피부 상태와 퍼스널 컬러를 반영한\n유쏘풀 향수 큐레이션</div>'
        '<div class="home-hero-desc">피부 상태를 먼저 확인하고 얼굴 사진으로 퍼스널 컬러를 분석해요. 그 결과를 바탕으로 어울리는 향과 유쏘풀 제품까지 차례로 추천해드릴게요.</div>'
        "</div>"
        '<div class="home-flow-head">이런 흐름으로 분석해요.</div>'
        f"{flow_markup}"
    )
    st.markdown(home_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    _, center, _ = st.columns([1.6, 1.5, 1.6])
    with center:
        if st.button("향수 추천 시작하기 →", type="primary", use_container_width=True):
            go_to("skin")


def render_integrated_feedback_section() -> None:
    ensure_integrated_feedback_state()
    predicted_label = st.session_state.personal_color
    class_options = list(load_personal_color_assets()["classes"])
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] label p {
            color:#8a776a !important;
            font-weight:700 !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.9) !important;
            border: 1px solid #ead8c9 !important;
            border-radius: 18px !important;
            color:#2f2622 !important;
            min-height: 54px !important;
        }
        div[data-testid="stSelectbox"] span {
            color:#2f2622 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    body_left, body_center, body_right = st.columns([0.6, 4.8, 0.6])
    with body_center:
        st.markdown("##### 예측 결과가 맞나요?")
        st.markdown(f"**{predicted_label}**")
        render_feedback_choice_buttons("feedback_match_answer", ["맞아요", "아니에요"], "feedback_match")

        is_match = st.session_state.feedback_match_answer == "맞아요"
        actual_label = predicted_label

        if not is_match:
            st.markdown("##### 실제 퍼스널 컬러를 알려주세요.")
            actual_label = st.selectbox(
                "실제 퍼스널 컬러",
                class_options,
                index=class_options.index(predicted_label) if predicted_label in class_options else 0,
                key="feedback_actual_label",
            )

        st.markdown(
            "##### 이 이미지를 학습 데이터로 사용해도 될까요?"
            if is_match
            else "##### 수정한 라벨과 함께 이 이미지를 학습과 검증에 사용해도 될까요?"
        )
        render_feedback_choice_buttons("feedback_consent_answer", ["동의", "비동의"], "feedback_consent")

        _, submit_col, _ = st.columns([1, 1.45, 1])
        with submit_col:
            if st.button("피드백 저장", type="primary", use_container_width=True, key="integrated_feedback_submit"):
                submit_integrated_feedback(
                    actual_label=actual_label,
                    is_match=is_match,
                    consent=st.session_state.feedback_consent_answer == "동의",
                )
                st.rerun()

    st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

    if st.session_state.feedback_saved and st.session_state.feedback_saved_message:
        st.success(st.session_state.feedback_saved_message)

def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return
    current = current_step_index()
    cols = st.columns(len(STEP_PAGES))
    for idx, (col, page) in enumerate(zip(cols, STEP_PAGES), start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="
                        width:50px;
                        height:50px;
                        margin:0 auto 0.6rem auto;
                        border-radius:999px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:{bubble_bg};
                        color:white;
                        font-weight:800;
                        font-size:1.25rem;
                    ">{bubble_text}</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{label_color};">{STEP_LABELS[page]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)


def render_loading_page() -> None:
    if st.session_state.uploaded_image is None:
        go_to("face")
        return
    st.markdown(
        """
        <style>
        .stButton,
        .stTabs,
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"],
        [data-testid="stImage"],
        .upload-preview,
        .upload-status,
        .hero-card,
        .season-card,
        .fragrance-card,
        .result-featured-product-card,
        .result-shell,
        .palette-card,
        .product-card {
            display:none !important;
        }
        .block-container {
            padding-top: 6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel" style="margin-top:1rem;">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1.18rem; font-weight:700; line-height:1.9; color:#675b54; max-width:260px;">
                업로드한 사진을 바탕으로 퍼스널 컬러를 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.analysis_pending:
        time.sleep(1.0)
        run_recommendation()
        st.session_state.analysis_pending = False
    st.session_state.page = "result"
    st.rerun()


def render_scent_loading_page() -> None:
    st.markdown(
        """
        <style>
        .stButton,
        .season-card,
        .fragrance-card,
        .result-featured-product-card,
        .result-shell,
        .palette-card,
        .product-card {
            display:none !important;
        }
        .block-container {
            padding-top: 6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel" style="margin-top:1rem;">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1.18rem; font-weight:700; line-height:1.9; color:#675b54; max-width:260px;">
                데이터 기반으로 나에게 어울리는 향을 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.scent_analysis_pending:
        time.sleep(1.1)
        st.session_state.scent_analysis_pending = False
    st.session_state.page = "scent_result"
    st.rerun()


def get_choice_display_content(state_key: str, index: int) -> Dict[str, str]:
    content_map = {
        "skin_type": [
            {"title": "지성", "hint": "유분이 비교적 많이 느껴지는 피부예요."},
            {"title": "건성", "hint": "건조함과 당김이 자주 느껴지는 피부예요."},
            {"title": "민감성", "hint": "자극과 붉어짐에 예민한 피부예요."},
            {"title": "잘 모르겠어요", "hint": "아직 피부 타입을 딱 정하기 어려운 편이에요."},
        ],
        "moisture_level": [
            {"title": "건조하지 않아요", "hint": "수분감이 비교적 유지되는 편이에요."},
            {"title": "건조해요", "hint": "수분이 쉽게 날아가고 메마름이 느껴져요."},
        ],
        "temperature_level": [
            {"title": "열감이 높은 편", "hint": "피부에 열감이 오래 남는 편이에요."},
            {"title": "열감이 낮은 편", "hint": "차분하고 서늘하게 느껴지는 편이에요."},
        ],
    }
    options = content_map.get(state_key, [])
    if 0 <= index < len(options):
        return options[index]
    return {"title": "", "hint": ""}


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    render_hero(kicker, title, description)
    st.markdown(
        """
        <style>
        .choice-shell {
            max-width: 760px;
            margin: 0 auto;
        }
        .choice-shell div.stButton > button {
            min-height: 66px !important;
        }
        .choice-hint {
            margin: 0.35rem 0 1.1rem 0.35rem;
            color: #8a776a;
            font-size: 0.94rem;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown('<div class="choice-shell">', unsafe_allow_html=True)
    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        content = get_choice_display_content(state_key, index)
        is_active = selected == item["key"]
        if st.button(
            content["title"] or item["title"],
            key=f"{state_key}_choice_final_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown(
            f'<div class="choice-hint">{content["hint"] or item["description"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-wrap-v2 {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-header-v2 {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-kicker-v2 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-title-v2 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-desc-v2 {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-row-v2 {
            margin-bottom: 0.5rem;
        }
        .choice-card-v2 {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #EADFD3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }
        .choice-card-v2.is-selected {
            border: 2px solid #5F4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #FFFAF5;
        }
        .choice-card-title-v2 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-card-desc-v2 {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
            flex: 1 1 auto;
        }
        .choice-card-desc-v2.is-selected {
            color: #8D7666;
        }
        .radio-col-v2 {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 72px;
            margin-bottom: 0.5rem;
        }
        .radio-col-v2 .stButton {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .radio-col-v2 .stButton > button {
            width: 44px !important;
            min-width: 44px !important;
            height: 44px !important;
            min-height: 44px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .radio-col-v2 .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 2px solid #D9CEC3 !important;
            color: transparent !important;
        }
        .radio-col-v2 .stButton > button[kind="primary"] {
            background: #9B7A67 !important;
            border: 2px solid #9B7A67 !important;
            color: #FFFFFF !important;
        }
        @media (max-width: 760px) {
            .choice-title-v2 { font-size: 2.25rem; }
            .choice-card-v2 { padding: 0.95rem 1.2rem; }
            .choice-card-title-v2 { font-size: 1.35rem; }
            .choice-card-desc-v2 { font-size: 0.92rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-wrap-v2">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-header-v2">
            <div class="choice-kicker-v2">{kicker}</div>
            <div class="choice-title-v2">{title}</div>
            <div class="choice-desc-v2">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)

    for index, item in enumerate(options):
        is_active = selected == item["key"]
        radio_col, card_col = st.columns([1, 9], gap="small")

        with radio_col:
            st.markdown('<div class="radio-col-v2">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_pick_v2_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with card_col:
            st.markdown(
                f"""
                <div class="choice-row-v2">
                    <div class="choice-card-v2{' is-selected' if is_active else ''}">
                        <div class="choice-card-title-v2">{item['title']}</div>
                        <div class="choice-card-desc-v2{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_v2", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_v2",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()
    render_step_progress()
    render_hero(
        "얼굴 사진 분석",
        "얼굴 사진으로\n퍼스널 컬러를 분석할게요.",
        "사진을 올리면 배경을 정리한 뒤 퍼스널 컬러를 분석해요.\n분석이 끝나면 결과 카드와 향 추천 화면으로 이어집니다.",
    )
    st.markdown(
        """
        <style>
        .face-upload-shell {
            max-width: 620px;
            margin: 0 auto;
        }
        .face-upload-shell .stTabs [data-baseweb="tab-list"] {
            justify-content: flex-end;
            margin-bottom: 1.05rem;
        }
        .face-upload-shell [data-testid="stFileUploader"],
        .face-upload-shell [data-testid="stCameraInput"] {
            width: 100%;
            margin-top: 0.7rem;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 251, 247, 0.96) !important;
            border: 1.4px dashed #d8bca5 !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.10) !important;
            min-height: 168px !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] section {
            background: transparent !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] * {
            color: #6f645b !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] svg {
            fill: #b48767 !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button {
            background: #a97757 !important;
            color: #ffffff !important;
            border: 1px solid #a97757 !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button:hover {
            background: #956848 !important;
            border-color: #956848 !important;
        }
        .face-upload-shell [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
        }
        .face-upload-shell [data-testid="stCameraInput"] > div {
            background: rgba(255, 251, 247, 0.96) !important;
            border: 1.4px dashed #d8bca5 !important;
            border-radius: 22px !important;
            box-shadow: 0 14px 28px rgba(199, 171, 145, 0.10) !important;
            padding: 0.85rem !important;
        }
        .face-upload-shell [data-testid="stCameraInput"] button {
            background: #fffaf6 !important;
            color: #9f6c45 !important;
            border: 1px solid #d8bca5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    _, center_area, _ = st.columns([1.1, 2.8, 1.1])
    with center_area:
        st.markdown('<div class="face-upload-shell">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["사진 업로드", "카메라로 찍기"])
        with tab1:
            uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        with tab2:
            camera_photo = st.camera_input("카메라 촬영")
            if camera_photo is not None:
                st.session_state.uploaded_image = camera_photo
                st.session_state.rembg_image = None
                st.session_state.prediction_error = None
        if st.session_state.uploaded_image is not None:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="업로드된 사진", width=520)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="upload-status">사진 업로드가 완료됐어요. 결과 보기를 누르면 분석이 시작됩니다.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        if st.button("이전", key="face_prev_final", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next_final", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def ensure_integrated_feedback_state() -> None:
    uploaded_name = getattr(st.session_state.uploaded_image, "name", "camera_capture")
    context_key = f"{uploaded_name}:{st.session_state.personal_color}"
    defaults = {
        "feedback_context_key": context_key,
        "feedback_match_answer": "맞아요",
        "feedback_actual_label": st.session_state.personal_color,
        "feedback_consent_answer": "비동의",
        "feedback_comment": "",
        "feedback_saved": False,
        "feedback_saved_message": None,
    }
    if st.session_state.get("feedback_context_key") != context_key:
        for key, value in defaults.items():
            st.session_state[key] = value
        return
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_integrated_feedback_log(
    *,
    uploaded_name: str,
    predicted_label: str,
    predicted_confidence: Optional[float],
    user_label: str,
    is_match: bool,
    consent: bool,
    feedback_comment: str = "",
    saved_image_path: str = "",
) -> None:
    ensure_feedback_dirs()
    is_new = not FEEDBACK_LOG_PATH.exists()
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "uploaded_name",
                "predicted_label",
                "predicted_confidence",
                "user_label",
                "is_match",
                "consent",
                "feedback_comment",
                "saved_image_path",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "uploaded_name": uploaded_name,
                "predicted_label": predicted_label,
                "predicted_confidence": f"{predicted_confidence:.4f}" if predicted_confidence is not None else "",
                "user_label": user_label,
                "is_match": "yes" if is_match else "no",
                "consent": "yes" if consent else "no",
                "feedback_comment": feedback_comment,
                "saved_image_path": saved_image_path,
            }
        )


def submit_integrated_feedback(actual_label: str, is_match: bool, consent: bool) -> None:
    uploaded_file = st.session_state.uploaded_image
    if uploaded_file is None:
        return
    saved_image_path = ""
    if consent:
        saved_image_path = save_integrated_feedback_image(uploaded_file, actual_label)
    save_integrated_feedback_log(
        uploaded_name=getattr(uploaded_file, "name", "camera_capture"),
        predicted_label=st.session_state.personal_color,
        predicted_confidence=st.session_state.prediction_confidence,
        user_label=actual_label,
        is_match=is_match,
        consent=consent,
        feedback_comment=st.session_state.get("feedback_comment", "").strip(),
        saved_image_path=saved_image_path,
    )
    st.session_state.feedback_saved = True
    if consent:
        st.session_state.feedback_saved_message = (
            "피드백과 이미지 사용 동의가 저장됐어요.\n"
            f"로그 저장 위치: {FEEDBACK_LOG_PATH}\n"
            f"이미지 저장 위치: {saved_image_path}"
        )
    else:
        st.session_state.feedback_saved_message = (
            "피드백은 저장했고, 이미지는 학습 데이터로 사용하지 않도록 기록해둘게요.\n"
            f"로그 저장 위치: {FEEDBACK_LOG_PATH}"
        )


def render_integrated_feedback_section() -> None:
    ensure_integrated_feedback_state()
    predicted_label = st.session_state.personal_color
    class_options = list(load_personal_color_assets()["classes"])
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stTextArea"] label p {
            color:#8a776a !important;
            font-weight:700 !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            background: rgba(255,251,247,0.96) !important;
            border: 1.4px solid #dcc7b6 !important;
            border-radius: 20px !important;
            color:#2f2622 !important;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 132px !important;
            box-shadow: 0 14px 28px rgba(199,171,145,0.10) !important;
            padding: 1rem 1.05rem !important;
            line-height: 1.7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    body_left, body_center, body_right = st.columns([0.45, 5.2, 0.45])
    with body_center:
        st.markdown("##### 예측 결과가 맞나요?")
        st.markdown(f"**{predicted_label}**")
        render_feedback_choice_buttons("feedback_match_answer", ["맞아요", "아니에요"], "feedback_match_latest")
        is_match = st.session_state.feedback_match_answer == "맞아요"
        actual_label = predicted_label
        if not is_match:
            st.markdown("##### 실제 퍼스널 컬러를 알려주세요.")
            actual_label = st.selectbox(
                "실제 퍼스널 컬러",
                class_options,
                index=class_options.index(predicted_label) if predicted_label in class_options else 0,
                key="feedback_actual_label_latest",
            )
            st.session_state.feedback_actual_label = actual_label
        st.markdown(
            "##### 이 이미지를 학습 데이터로 사용해도 될까요?"
            if is_match
            else "##### 수정한 라벨과 함께 이 이미지를 학습과 검증에 사용해도 될까요?"
        )
        render_feedback_choice_buttons("feedback_consent_answer", ["동의", "비동의"], "feedback_consent_latest")
        st.markdown("##### 어떤 점이 부족했는지, 추가됐으면 하는 점이 있다면 적어주세요.")
        st.text_area(
            "추가 피드백",
            key="feedback_comment",
            placeholder="예: 설명이 더 자세했으면 좋겠어요. 추천 향 이유를 더 보고 싶어요.",
            height=132,
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        _, submit_col, _ = st.columns([1, 1.6, 1])
        with submit_col:
            if st.button("피드백 저장", type="primary", use_container_width=True, key="integrated_feedback_submit_latest"):
                submit_integrated_feedback(
                    actual_label=actual_label,
                    is_match=is_match,
                    consent=st.session_state.feedback_consent_answer == "동의",
                )
                st.rerun()
    st.markdown("<div style='height:4rem;'></div>", unsafe_allow_html=True)
    if st.session_state.feedback_saved and st.session_state.feedback_saved_message:
        st.success(st.session_state.feedback_saved_message)


def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return
    st.markdown("<div style='height:2.8rem;'></div>", unsafe_allow_html=True)
    current = current_step_index()
    cols = st.columns(len(STEP_PAGES))
    for idx, (col, page) in enumerate(zip(cols, STEP_PAGES), start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="
                        width:50px;
                        height:50px;
                        margin:0 auto 0.55rem auto;
                        border-radius:999px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:{bubble_bg};
                        color:white;
                        font-weight:800;
                        font-size:1.25rem;
                    ">{bubble_text}</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{label_color};">{STEP_LABELS[page]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:3.2rem;'></div>", unsafe_allow_html=True)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-page-wrap {
            max-width: 760px;
            margin: 0 auto;
        }
        .choice-page-wrap .hero-card {
            text-align: center;
            padding: 2.2rem 2.2rem 2rem 2.2rem;
        }
        .choice-page-wrap .hero-card > div:first-child,
        .choice-page-wrap .hero-card > div:nth-child(2),
        .choice-page-wrap .hero-card > div:nth-child(3) {
            text-align: center;
        }
        .choice-layout-card {
            background: rgba(255, 252, 248, 0.96);
            border: 1.35px solid #e2cdbc;
            border-radius: 22px;
            padding: 1rem 1.15rem 0.95rem 1.15rem;
            box-shadow: 0 12px 26px rgba(199, 171, 145, 0.08);
            margin-bottom: 0.5rem;
        }
        .choice-layout-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #2f2622;
            margin-bottom: 0.35rem;
        }
        .choice-layout-copy {
            color: #7d7068;
            font-size: 0.97rem;
            line-height: 1.68;
        }
        .choice-page-wrap div.stButton > button {
            min-height: 42px !important;
            font-size: 0.98rem !important;
        }
        .choice-action-row {
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="choice-page-wrap">', unsafe_allow_html=True)
    render_hero(kicker, title, description)
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        st.markdown(
            f"""
            <div class="choice-layout-card" style="border-color:{'#8c63ff' if is_active else '#e2cdbc'};">
                <div class="choice-layout-title">{item['title']}</div>
                <div class="choice-layout-copy">{item['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="choice-action-row">', unsafe_allow_html=True)
        if st.button(
            item["button"],
            key=f"{state_key}_layout_pick_final_{index}",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("← 이전", key=f"prev_{state_key}_layout_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음 단계로 →",
            key=f"next_{state_key}_layout_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()
    render_step_progress()
    st.markdown(
        """
        <style>
        .face-page-wrap {
            max-width: 780px;
            margin: 0 auto;
        }
        .face-page-header {
            text-align: center;
            margin-bottom: 1.4rem;
        }
        .face-page-title {
            font-size: 2.9rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.03em;
        }
        .face-page-desc {
            margin-top: 0.85rem;
            color: #7c6d63;
            font-size: 1.05rem;
            line-height: 1.8;
        }
        .face-tip-box {
            background: rgba(255,252,248,0.96);
            border: 1.4px solid #f2a3a0;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 1.65rem;
            box-shadow: 0 10px 24px rgba(199,171,145,0.08);
        }
        .face-tip-head {
            padding: 0.95rem 1.1rem;
            font-weight: 800;
            color: #5e4a42;
            border-bottom: 1px solid #ead8c9;
            background: rgba(255,247,246,0.85);
        }
        .face-tip-list {
            margin: 0;
            padding: 1rem 1.45rem 1.05rem 2rem;
            color: #6f645b;
            line-height: 1.9;
            font-size: 0.98rem;
        }
        .face-page-wrap .stTabs [data-baseweb="tab-list"] {
            gap: 0.7rem;
            border-bottom: 1px solid #e4d4c8;
            margin-bottom: 1rem;
        }
        .face-page-wrap [data-testid="stFileUploader"],
        .face-page-wrap [data-testid="stCameraInput"] {
            width: 100%;
        }
        .face-page-wrap [data-testid="stFileUploaderDropzone"] {
            background: #eef1f5 !important;
            border: none !important;
            border-radius: 18px !important;
            min-height: 118px !important;
            box-shadow: none !important;
        }
        .face-page-wrap [data-testid="stFileUploaderDropzone"] section {
            background: transparent !important;
        }
        .face-page-wrap [data-testid="stFileUploaderDropzone"] * {
            color: #6f645b !important;
        }
        .face-page-wrap [data-testid="stFileUploaderDropzone"] svg {
            fill: #9fb1cc !important;
        }
        .face-page-wrap [data-testid="stFileUploaderDropzone"] button {
            background: #fffaf6 !important;
            color: #9f6c45 !important;
            border: 1px solid #d8bca5 !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        .face-page-wrap [data-testid="stCameraInput"] > div {
            background: #eef1f5 !important;
            border: none !important;
            border-radius: 18px !important;
            box-shadow: none !important;
            padding: 0.65rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="face-page-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="face-page-header">
            <div class="face-page-title">얼굴 사진을 올려주세요</div>
            <div class="face-page-desc">AI가 퍼스널 컬러를 분석해 향수를 추천해드려요</div>
        </div>
        <div class="face-tip-box">
            <div class="face-tip-head">사진 잘 찍는 팁 (더 정확한 분석을 위해)</div>
            <ul class="face-tip-list">
                <li>자연광이나 흰 조명, 무채색 배경이 가장 좋아요!</li>
                <li>이목구비 (눈, 코, 입)와 머리카락이 전부 보이는 정면 사진을 사용해주세요.</li>
                <li>쇄골 위로 두상만 나올 수 있게 촬영해주세요.</li>
                <li>모자, 안경, 선글라스, 마스크 등은 벗어주세요.</li>
                <li>화장을 하지 않은 상태에서의 진단이 가장 정확해요.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["사진 업로드", "카메라로 찍기"])
    with tab1:
        uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            st.session_state.uploaded_image = uploaded
            st.session_state.rembg_image = None
            st.session_state.prediction_error = None
    with tab2:
        camera_photo = st.camera_input("카메라 촬영")
        if camera_photo is not None:
            st.session_state.uploaded_image = camera_photo
            st.session_state.rembg_image = None
            st.session_state.prediction_error = None
    if st.session_state.uploaded_image is not None:
        st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image, caption="업로드된 사진", width=520)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="upload-status">사진 업로드가 완료됐어요. 결과 보기를 누르면 분석이 시작됩니다.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("← 이전", key="face_prev_layout_final", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next_layout_final", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return
    st.markdown("<div style='height:4.3rem;'></div>", unsafe_allow_html=True)
    current = current_step_index()
    progress_markup = []
    for idx, page in enumerate(STEP_PAGES, start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)
        progress_markup.append(
            f"""
            <div style="text-align:center; min-width:72px;">
                <div style="
                    width:50px;
                    height:50px;
                    margin:0 auto 0.6rem auto;
                    border-radius:999px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:{bubble_bg};
                    color:white;
                    font-weight:800;
                    font-size:1.2rem;
                ">{bubble_text}</div>
                <div style="font-size:0.95rem; font-weight:700; color:{label_color}; white-space:nowrap;">{STEP_LABELS[page]}</div>
            </div>
            """
        )
        if idx < len(STEP_PAGES):
            progress_markup.append(
                """
                <div style="
                    display:flex;
                    align-items:flex-start;
                    justify-content:center;
                    min-width:34px;
                    padding-top:0.82rem;
                    color:#b7a79a;
                    font-weight:800;
                    font-size:1.1rem;
                    letter-spacing:0.18rem;
                ">· · ·</div>
                """
            )
    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:center;
            align-items:flex-start;
            gap:0.18rem;
            flex-wrap:nowrap;
            width:100%;
            overflow-x:auto;
            padding-bottom:0.2rem;
        ">
            {''.join(progress_markup)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:3.7rem;'></div>", unsafe_allow_html=True)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-page-wrap-final {
            max-width: 790px;
            margin: 0 auto;
        }
        .choice-page-header-final {
            text-align: center;
            margin-bottom: 2.1rem;
        }
        .choice-page-kicker-final {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-page-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-page-desc-final {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-info-card-final {
            background: rgba(253, 250, 246, 0.98);
            border: 1.35px solid #dfc7b2;
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 12px 28px rgba(199, 171, 145, 0.08);
            margin-bottom: 0.65rem;
        }
        .choice-info-card-final.is-selected {
            background: #fff6f0;
            border-color: #b9835b;
        }
        .choice-info-row-final {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            flex-wrap: nowrap;
        }
        .choice-info-title-final {
            font-size: 1.55rem;
            font-weight: 800;
            color: #2f2622;
            margin-bottom: 0;
            flex: 0 0 auto;
            white-space: nowrap;
        }
        .choice-info-copy-final {
            color: #776a62;
            font-size: 0.98rem;
            line-height: 1.65;
            flex: 1 1 auto;
        }
        @media (max-width: 720px) {
            .choice-info-row-final {
                flex-wrap: wrap;
                gap: 0.35rem;
            }
            .choice-info-title-final,
            .choice-info-copy-final {
                width: 100%;
            }
        }
        .choice-action-row-final {
            margin: 0.72rem 0 1.2rem 0;
        }
        .choice-page-wrap-final div.stButton > button {
            min-height: 54px !important;
            border-radius: 14px !important;
            font-size: 1rem !important;
        }
        .choice-button-selected-final div.stButton > button {
            background: #b9835b !important;
            border-color: #b9835b !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="choice-page-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-page-header-final">
            <div class="choice-page-kicker-final">{kicker}</div>
            <div class="choice-page-title-final">{title}</div>
            <div class="choice-page-desc-final">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        st.markdown(
            f"""
            <div class="choice-info-card-final{' is-selected' if is_active else ''}">
                <div class="choice-info-row-final">
                    <div class="choice-info-title-final">{item['title']}</div>
                    <div class="choice-info-copy-final">{item['description']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="choice-action-row-final{" choice-button-selected-final" if is_active else ""}">',
            unsafe_allow_html=True,
        )
        if st.button(
            item["button"],
            key=f"{state_key}_layout_pick_override_{index}",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_layout_override", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_layout_override",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_face_page() -> None:
    if st.session_state.analysis_pending:
        st.session_state.page = "loading"
        st.rerun()
    render_step_progress()
    st.markdown(
        """
        <style>
        .face-page-wrap-final {
            max-width: 810px;
            margin: 0 auto;
        }
        .face-page-header-final {
            text-align: center;
            margin-bottom: 1.55rem;
        }
        .face-page-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            margin-bottom: 0.95rem;
        }
        .face-page-desc-final {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
        }
        .face-tip-wrap-final {
            margin: 1.85rem 0 1.9rem 0;
        }
        .face-page-wrap-final details {
            background: rgba(253, 250, 246, 0.98);
            border: 1.45px solid #f0a4a1;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(199, 171, 145, 0.08);
        }
        .face-page-wrap-final summary {
            list-style: none;
            cursor: pointer;
            padding: 1rem 1.15rem;
            font-weight: 800;
            color: #5e4a42;
            background: rgba(255, 247, 246, 0.85);
            border-bottom: 1px solid #ecd8cd;
        }
        .face-page-wrap-final summary::-webkit-details-marker {
            display: none;
        }
        .face-tip-list-final {
            margin: 0;
            padding: 1rem 1.5rem 1.1rem 2.2rem;
            color: #6f645b;
            line-height: 1.9;
            font-size: 1rem;
        }
        .face-page-wrap-final .stTabs [data-baseweb="tab-list"] {
            gap: 0.7rem;
            border-bottom: 1px solid #e4d4c8;
            justify-content: flex-start;
            margin-bottom: 1.35rem;
        }
        .face-page-wrap-final .stTabs [data-baseweb="tab"] {
            padding: 0.12rem 0.42rem 1.35rem 0.42rem;
        }
        .face-page-wrap-final [data-testid="stFileUploader"],
        .face-page-wrap-final [data-testid="stCameraInput"] {
            width: 100%;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] {
            background: #eef1f5 !important;
            border: none !important;
            border-radius: 18px !important;
            min-height: 118px !important;
            box-shadow: none !important;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] section {
            background: transparent !important;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] * {
            color: #6f645b !important;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] svg {
            fill: #9fb1cc !important;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button {
            background: #a97757 !important;
            color: #ffffff !important;
            border: 1px solid #a97757 !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button:hover {
            background: #946646 !important;
            border-color: #946646 !important;
        }
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button,
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button *,
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button span,
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button p,
        .face-page-wrap-final [data-testid="stFileUploaderDropzone"] button div {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        .face-page-wrap-final [data-testid="stCameraInput"] > div {
            background: #eef1f5 !important;
            border: none !important;
            border-radius: 18px !important;
            box-shadow: none !important;
            padding: 0.8rem !important;
        }
        .face-preview-wrap-final {
            margin-top: 1.15rem;
            text-align: center;
        }
        .face-upload-status-final {
            margin-top: 0.8rem;
            color: #8a7669;
            font-size: 0.95rem;
            line-height: 1.7;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="face-page-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="face-page-header-final">
            <div class="face-page-title-final">얼굴 사진을 올려주세요</div>
            <div class="face-page-desc-final">AI가 퍼스널 컬러를 분석해 향수를 추천해드려요</div>
        </div>
        <div class="face-tip-wrap-final">
            <details open>
                <summary>📌 사진 잘 찍는 팁 (더 정확한 분석을 위해)</summary>
                <ul class="face-tip-list-final">
                    <li>자연광이나 흰 조명, 무채색 배경이 가장 좋아요!</li>
                    <li>이목구비 (눈, 코, 입)와 머리카락이 전부 보이는 정면 사진을 사용해주세요.</li>
                    <li>쇄골 위로 두상만 나올 수 있게 촬영해주세요.</li>
                    <li>모자, 안경, 선글라스, 마스크 등은 벗어주세요.</li>
                    <li>화장을 하지 않은 상태에서의 진단이 가장 정확해요.</li>
                </ul>
            </details>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["사진 업로드", "카메라로 찍기"])
    with tab1:
        uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.uploaded_image = uploaded
            st.session_state.rembg_image = None
            st.session_state.prediction_error = None
    with tab2:
        camera_photo = st.camera_input("카메라 촬영", label_visibility="collapsed")
        if camera_photo is not None:
            st.session_state.uploaded_image = camera_photo
            st.session_state.rembg_image = None
            st.session_state.prediction_error = None
    if st.session_state.uploaded_image is not None:
        preview_left, preview_center, preview_right = st.columns([0.9, 1.5, 0.9])
        with preview_center:
            st.markdown('<div class="face-preview-wrap-final">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="업로드한 사진", width=520)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="face-upload-status-final">사진 업로드가 완료되었어요. 결과 보기를 누르면 분석이 시작됩니다.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key="face_prev_layout_override", use_container_width=True):
            go_to("temperature")
    with right:
        if st.button("결과 보기 →", key="face_next_layout_override", type="primary", use_container_width=True):
            st.session_state.analysis_pending = True
            st.session_state.page = "loading"
            st.rerun()


def render_loading_page() -> None:
    if st.session_state.uploaded_image is None:
        go_to("face")
        return
    st.markdown(
        """
        <style>
        .stButton,
        .stTabs,
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"],
        [data-testid="stImage"],
        .upload-preview,
        .upload-status,
        .hero-card,
        .season-card,
        .fragrance-card,
        .result-featured-product-card,
        .result-shell,
        .palette-card,
        .product-card,
        .face-page-wrap,
        .face-page-wrap-final,
        .face-page-header,
        .face-page-header-final,
        .face-tip-box,
        .face-tip-wrap,
        .face-tip-wrap-final,
        details,
        summary,
        .face-upload-status-final,
        .face-preview-wrap-final {
            display: none !important;
        }
        .block-container {
            padding-top: 6.8rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel" style="margin-top:1.6rem;">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1.28rem; font-weight:800; line-height:1.9; color:#675b54; max-width:270px;">
                업로드한 사진을 바탕으로 퍼스널 컬러를 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.analysis_pending:
        time.sleep(1.0)
        run_recommendation()
        st.session_state.analysis_pending = False
    st.session_state.page = "result"
    st.rerun()


def render_scent_loading_page() -> None:
    st.markdown(
        """
        <style>
        .stButton,
        .season-card,
        .fragrance-card,
        .result-featured-product-card,
        .result-shell,
        .palette-card,
        .product-card {
            display:none !important;
        }
        .block-container {
            padding-top: 6.8rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dots = []
    radius = 38
    for index in range(10):
        angle = (index / 10) * 2 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        dots.append(
            f'<div class="loading-dot" style="transform: translate({x:.1f}px, {y:.1f}px); animation-delay:{index * 0.09:.2f}s;"></div>'
        )
    st.markdown(
        f"""
        <div class="loading-panel" style="margin-top:1.6rem;">
            <div class="loading-ring">{''.join(dots)}</div>
            <div style="font-size:1.28rem; font-weight:800; line-height:1.9; color:#675b54; max-width:270px;">
                데이터 기반으로 나에게 어울리는 향을 분석 중입니다...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(0.9)
    st.session_state.page = "scent_result"
    st.rerun()


def render_signature_palette_card(personal_color: str) -> None:
    profile = PERSONAL_COLOR_PROFILES.get(personal_color)
    palette = PERSONAL_COLOR_PALETTES.get(personal_color)
    palette_image_path = PERSONAL_COLOR_PALETTE_IMAGES.get(personal_color)
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_path = PERSONAL_COLOR_REPRESENTATIVE_IMAGES.get(personal_color)

    if not profile or not palette:
        return

    image_data = image_to_base64(image_path) if image_path else None
    image_markup = ""
    if image_data:
        image_markup = (
            f'<img class="season-avatar" src="data:image/png;base64,{image_data}" alt="{personal_color} 대표 이미지" />'
        )

    keyword_pills = "".join(
        f'<span class="season-pill" style="background:{theme["accent"]};">{keyword}</span>'
        for keyword in profile.get("keywords", [])
    )
    description_lines = "<br>".join(profile.get("description_lines", []))
    colors = palette.get("colors", [])
    color_names = [color.get("name", "").strip() for color in colors if color.get("name", "").strip()]
    if not color_names:
        color_names = [keyword.replace("#", "") for keyword in profile.get("keywords", []) if keyword]
    palette_caption = " / ".join(color_names[:4])

    swatch_markup = "".join(
        f"""
        <div class="season-swatch-item" style="width:100%; min-width:0;">
            <div class="season-swatch-circle" style="background:{color['hex']}; margin:0 auto;"></div>
            <div class="season-swatch-name">{color['name']}</div>
        </div>
        """
        for color in colors
    )

    if colors:
        palette_markup = (
            '<div style="width:100%; max-width:100%; margin:0 auto;">'
            '<div style="height:2px; background:rgba(60, 52, 48, 0.88); width:96%; margin:0 auto 1rem auto;"></div>'
            f'<div class="season-subtitle" style="color:{theme["accent"]}; margin-bottom:1rem;">어울리는 색상</div>'
            f'<div class="season-swatch-grid" style="width:92%; max-width:92%; margin:0 auto; display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); column-gap:1.25rem; row-gap:1.25rem;">{swatch_markup}</div>'
            '</div>'
        )
    else:
        palette_markup = ""
        if palette_image_path and palette_image_path.exists():
            palette_image_data = image_to_base64(palette_image_path)
            if palette_image_data:
                palette_markup = (
                    '<div style="width:100%; max-width:100%; margin:0 auto;">'
                    '<div style="height:2px; background:rgba(60, 52, 48, 0.88); width:96%; margin:0 auto 1rem auto;"></div>'
                    f'<div class="season-subtitle" style="color:{theme["accent"]}; margin-bottom:1rem;">어울리는 색상</div>'
                    f'<div style="width:92%; margin:0 auto 1.1rem auto;">'
                    f'<img src="data:image/png;base64,{palette_image_data}" alt="{personal_color} 팔레트" style="width:100%; display:block; border-radius:28px;" />'
                    f'<div class="season-swatch-name" style="margin-top:0.7rem; text-align:center;">{palette_caption}</div>'
                    f"</div></div>"
                )

    st.markdown(
        f"""
        <div class="season-card" style="
            border:1.6px solid {hex_to_rgba(theme['accent'], 0.72)};
            box-shadow: 0 24px 55px {hex_to_rgba(theme['accent'], 0.16)};
        ">
            {image_markup}
            <div class="season-title" style="color:{theme['accent']};">{personal_color}</div>
            <div class="season-pill-row">{keyword_pills}</div>
            <div class="season-body">{description_lines}</div>
            {palette_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color

    render_signature_palette_card(selected_color)

    st.markdown("<div style='height:2.35rem;'></div>", unsafe_allow_html=True)
    prev_col, restart_col, scent_col = st.columns([1, 1, 1], gap="medium")
    with prev_col:
        if st.button("이전", key="result_prev_inline", use_container_width=True):
            go_to("face")
    with restart_col:
        if st.button("처음부터 다시", key="restart_inline", use_container_width=True):
            reset_all()
            st.rerun()
    with scent_col:
        if st.button("나에게 어울리는 향 찾기 →", key="to_scent_result_inline", type="primary", use_container_width=True):
            st.session_state.scent_analysis_pending = True
            st.session_state.page = "scent_loading"
            st.rerun()


PAGE_PROGRESS_ALIAS["product_feedback"] = "result"


def render_step_progress() -> None:
    progress_page = PAGE_PROGRESS_ALIAS.get(st.session_state.page, st.session_state.page)
    if progress_page not in STEP_PAGES:
        return

    st.markdown("<div style='height:4.3rem;'></div>", unsafe_allow_html=True)
    current = current_step_index()
    column_widths: List[float] = []
    for idx in range(len(STEP_PAGES)):
        column_widths.append(1.0)
        if idx < len(STEP_PAGES) - 1:
            column_widths.append(0.46)
    cols = st.columns(column_widths, gap="small")

    cursor = 0
    for idx, page in enumerate(STEP_PAGES, start=1):
        if idx - 1 < current:
            bubble_bg = "#c8b8a9"
            label_color = "#85776d"
            bubble_text = "✓"
        elif idx - 1 == current:
            bubble_bg = "#a87a5b"
            label_color = "#2f2622"
            bubble_text = str(idx)
        else:
            bubble_bg = "#d8ccc1"
            label_color = "#85776d"
            bubble_text = str(idx)

        with cols[cursor]:
            st.markdown(
                (
                    f'<div style="text-align:center;">'
                    f'<div style="width:50px;height:50px;margin:0 auto 0.6rem auto;'
                    f'border-radius:999px;display:flex;align-items:center;justify-content:center;'
                    f'background:{bubble_bg};color:white;font-weight:800;font-size:1.2rem;">{bubble_text}</div>'
                    f'<div style="font-size:0.95rem;font-weight:700;color:{label_color};white-space:nowrap;">{STEP_LABELS[page]}</div>'
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )
        cursor += 1

        if idx < len(STEP_PAGES):
            with cols[cursor]:
                st.markdown(
                    '<div style="text-align:center;padding-top:0.84rem;color:#b7a79a;font-weight:800;font-size:1.02rem;letter-spacing:0.06rem;white-space:nowrap;line-height:1;">···</div>',
                    unsafe_allow_html=True,
                )
            cursor += 1
    st.markdown("<div style='height:3.7rem;'></div>", unsafe_allow_html=True)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .skin-choice-wrap-final {
            max-width: 820px;
            margin: 0 auto;
        }
        .skin-choice-header-final {
            text-align: center;
            margin-bottom: 2.2rem;
        }
        .skin-choice-kicker-final {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .skin-choice-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .skin-choice-desc-final {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .skin-choice-row-final {
            margin-bottom: 1rem;
        }
        .skin-choice-text-box-final {
            background: rgba(255, 252, 248, 0.98);
            border: 1.3px solid #eadfd3;
            border-radius: 999px;
            min-height: 72px;
            display:flex;
            align-items:center;
            padding: 0.75rem 3.8rem 0.75rem 1.6rem;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
        }
        .skin-choice-text-box-final.is-selected {
            border: 2px solid #2f2622;
            box-shadow: 0 10px 20px rgba(35, 31, 29, 0.08);
        }
        .skin-check-wrap-final {
            display:flex;
            align-items:center;
            justify-content:flex-end;
            height:72px;
        }
        .skin-check-wrap-final .stButton {
            display:flex !important;
            justify-content:flex-end !important;
            margin-left:-3.25rem !important;
            z-index:3;
        }
        .skin-check-wrap-final .stButton > button {
            min-height: 0 !important;
            height: 42px !important;
            width: 42px !important;
            min-width: 42px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            font-size: 1rem !important;
            margin: 0 !important;
        }
        .skin-check-wrap-final .stButton > button[kind="secondary"] {
            background: white !important;
            border: 1.5px solid #d8cec5 !important;
        }
        .skin-check-wrap-final .stButton > button[kind="primary"] {
            background: #9b7a67 !important;
            border: 1.5px solid #9b7a67 !important;
        }
        .skin-choice-row-title-final {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
        }
        .skin-choice-row-copy-final {
            font-size: 1rem;
            color: #b9ada5;
            line-height: 1.6;
        }
        .skin-choice-row-copy-final.is-selected {
            color: #9a7b67;
        }
        .skin-choice-text-wrap-final {
            display:flex;
            align-items:center;
            gap:1rem;
            width:100%;
        }
        @media (max-width: 720px) {
            .skin-choice-row-title-final {
                font-size: 1.35rem;
            }
            .skin-choice-row-copy-final {
                font-size: 0.92rem;
            }
            .skin-choice-text-wrap-final {
                gap: 0.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="skin-choice-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="skin-choice-header-final">
            <div class="skin-choice-kicker-final">{kicker}</div>
            <div class="skin-choice-title-final">{title}</div>
            <div class="skin-choice-desc-final">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        st.markdown('<div class="skin-choice-row-final">', unsafe_allow_html=True)
        text_col, button_col = st.columns([14.0, 0.85], gap="small")
        with text_col:
            st.markdown(
                f"""
                <div class="skin-choice-text-box-final{' is-selected' if is_active else ''}">
                    <div class="skin-choice-text-wrap-final">
                        <div class="skin-choice-row-title-final">{item['title']}</div>
                        <div class="skin-choice-row-copy-final{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with button_col:
            st.markdown('<div class="skin-check-wrap-final">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_selector_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_skin_design", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_skin_design",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_scent_result_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color
    products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    recommended_labels = st.session_state.recommended_labels

    render_fragrance_section(selected_color, recommended_labels, products)

    st.markdown("<div style='height:2.35rem;'></div>", unsafe_allow_html=True)
    prev_col, restart_col, next_col = st.columns([1, 1, 1], gap="medium")
    with prev_col:
        if st.button("이전 결과", key="back_to_result_split", use_container_width=True):
            go_to("result")
    with restart_col:
        if st.button("처음부터 다시", key="restart_scent_split", use_container_width=True):
            reset_all()
            st.rerun()
    with next_col:
        if st.button("유쏘풀 추천 향수 보기 →", key="to_product_feedback", type="primary", use_container_width=True):
            st.session_state.page = "product_feedback"
            st.rerun()


def render_product_feedback_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color
    products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    recommended_labels = st.session_state.recommended_labels

    featured_product = pick_featured_product(products)
    if featured_product:
        render_native_featured_product_card(selected_color, featured_product)
        render_share_actions(selected_color, recommended_labels, featured_product)

    render_integrated_feedback_section()

    st.markdown("<div style='height:2.1rem;'></div>", unsafe_allow_html=True)
    left_btn, center_btn, right_btn = st.columns(3)
    with left_btn:
        if st.button("이전", key="back_to_scent_result_split", use_container_width=True):
            go_to("scent_result")
    with center_btn:
        if st.button("사진 다시 선택", key="back_to_face_split", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart_product_feedback", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


def main() -> None:
    init_session()
    if st.session_state.analysis_pending and st.session_state.page == "face":
        st.session_state.page = "loading"
    if st.session_state.scent_analysis_pending and st.session_state.page == "result":
        st.session_state.page = "scent_loading"
    apply_global_style()
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "skin":
        render_skin_choice_page(
            "피부 상태 체크",
            "지금 내 피부에\n가장 가까운 타입을 골라주세요",
            "평소 느끼는 피부 상태와 가장 가까운 항목 하나를 선택해 주세요.\n이 선택이 이후 향 추천의 기준이 됩니다.",
            SKIN_OPTIONS,
            "skin_type",
            "home",
            "moisture",
        )
    elif st.session_state.page == "moisture":
        render_choice_page(
            "수분감 체크",
            "평소 피부에 남는\n수분감을 골라주세요",
            "피부가 촉촉한 편인지, 건조한 편인지 골라주세요.\n향이 피부에 머무는 느낌과 지속감을 맞출 때 함께 참고해요.",
            MOISTURE_OPTIONS,
            "moisture_level",
            "skin",
            "temperature",
        )
    elif st.session_state.page == "temperature":
        render_choice_page(
            "피부 온도 체크",
            "피부 열감과 온도감도\n함께 골라주세요",
            "열감이 자주 도는 편인지, 차분하고 서늘한 편인지 선택해 주세요.\n발향 인상과 잔향 무드를 조정할 때 반영됩니다.",
            TEMPERATURE_OPTIONS,
            "temperature_level",
            "moisture",
            "face",
        )
    elif st.session_state.page == "face":
        render_face_page()
    elif st.session_state.page == "loading":
        render_loading_page()
    elif st.session_state.page == "result":
        render_result_page()
    elif st.session_state.page == "scent_loading":
        render_scent_loading_page()
    elif st.session_state.page == "scent_result":
        render_scent_result_page()
    elif st.session_state.page == "product_feedback":
        render_product_feedback_page()


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .skin-choice-wrap-final2 {
            max-width: 820px;
            margin: 0 auto;
        }
        .skin-choice-header-final2 {
            text-align: center;
            margin-bottom: 2.2rem;
        }
        .skin-choice-kicker-final2 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .skin-choice-title-final2 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .skin-choice-desc-final2 {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .skin-card-box-final2 {
            background: rgba(255, 252, 248, 0.98);
            border: 1.3px solid #eadfd3;
            border-radius: 999px;
            min-height: 72px;
            display:flex;
            align-items:center;
            padding: 0.75rem 4.2rem 0.75rem 1.6rem;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
        }
        .skin-card-box-final2.is-selected {
            border: 2px solid #2f2622;
            box-shadow: 0 10px 20px rgba(35, 31, 29, 0.08);
        }
        .skin-card-text-final2 {
            display:flex;
            align-items:center;
            gap:1rem;
            width:100%;
        }
        .skin-card-title-final2 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .skin-card-copy-final2 {
            font-size: 1rem;
            color: #b9ada5;
            line-height: 1.6;
            flex: 1 1 auto;
        }
        .skin-card-copy-final2.is-selected {
            color: #9a7b67;
        }
        .skin-check-slot-final2 {
            display:flex;
            align-items:center;
            justify-content:flex-end;
            min-height:72px;
        }
        .skin-check-slot-final2 .stButton {
            display:flex !important;
            justify-content:flex-end !important;
            margin-left:-4.2rem !important;
            z-index:5;
        }
        .skin-check-slot-final2 .stButton > button {
            width:42px !important;
            min-width:42px !important;
            height:42px !important;
            min-height:42px !important;
            border-radius:999px !important;
            padding:0 !important;
            margin:0 !important;
            box-shadow:none !important;
            font-size:1rem !important;
        }
        .skin-check-slot-final2 .stButton > button[kind="secondary"] {
            background:#ffffff !important;
            border:1.5px solid #d8cec5 !important;
            color:transparent !important;
        }
        .skin-check-slot-final2 .stButton > button[kind="primary"] {
            background:#9b7a67 !important;
            border:1.5px solid #9b7a67 !important;
            color:#ffffff !important;
        }
        @media (max-width: 720px) {
            .skin-card-box-final2 {
                padding-right: 3.5rem;
            }
            .skin-card-text-final2 {
                gap: 0.55rem;
                flex-wrap: wrap;
            }
            .skin-card-title-final2 {
                font-size: 1.35rem;
                width: 100%;
            }
            .skin-card-copy-final2 {
                font-size: 0.92rem;
                width: 100%;
            }
            .skin-check-slot-final2 .stButton {
                margin-left: -3.6rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="skin-choice-wrap-final2">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="skin-choice-header-final2">
            <div class="skin-choice-kicker-final2">{kicker}</div>
            <div class="skin-choice-title-final2">{title}</div>
            <div class="skin-choice-desc-final2">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        text_col, check_col = st.columns([13.6, 0.7], gap="small")
        with text_col:
            st.markdown(
                f"""
                <div class="skin-card-box-final2{' is-selected' if is_active else ''}">
                    <div class="skin-card-text-final2">
                        <div class="skin-card-title-final2">{item['title']}</div>
                        <div class="skin-card-copy-final2{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with check_col:
            st.markdown('<div class="skin-check-slot-final2">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_selector_fix_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_skin_design_fix", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_skin_design_fix",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-inline-wrap-final {
            max-width: 820px;
            margin: 0 auto;
        }
        .choice-inline-header-final {
            text-align: center;
            margin-bottom: 2.2rem;
        }
        .choice-inline-kicker-final {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-inline-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-inline-desc-final {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-inline-card-final {
            background: rgba(255, 252, 248, 0.98);
            border: 1.3px solid #eadfd3;
            border-radius: 999px;
            min-height: 72px;
            display:flex;
            align-items:center;
            padding: 0.75rem 4.2rem 0.75rem 1.6rem;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            margin-bottom: 1rem;
        }
        .choice-inline-card-final.is-selected {
            border: 2px solid #2f2622;
            box-shadow: 0 10px 20px rgba(35, 31, 29, 0.08);
        }
        .choice-inline-text-final {
            display:flex;
            align-items:center;
            gap:1rem;
            width:100%;
        }
        .choice-inline-title-text-final {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-inline-copy-final {
            font-size: 1rem;
            color: #b9ada5;
            line-height: 1.6;
            flex: 1 1 auto;
        }
        .choice-inline-copy-final.is-selected {
            color: #9a7b67;
        }
        .choice-inline-check-final {
            display:flex;
            align-items:center;
            justify-content:flex-end;
            min-height:72px;
        }
        .choice-inline-check-final .stButton {
            display:flex !important;
            justify-content:flex-end !important;
            margin-left:-4.25rem !important;
            z-index:5;
        }
        .choice-inline-check-final .stButton > button {
            width:42px !important;
            min-width:42px !important;
            height:42px !important;
            min-height:42px !important;
            border-radius:999px !important;
            padding:0 !important;
            margin:0 !important;
            box-shadow:none !important;
            font-size:1rem !important;
        }
        .choice-inline-check-final .stButton > button[kind="secondary"] {
            background:#ffffff !important;
            border:1.5px solid #d8cec5 !important;
            color:transparent !important;
        }
        .choice-inline-check-final .stButton > button[kind="primary"] {
            background:#9b7a67 !important;
            border:1.5px solid #9b7a67 !important;
            color:#ffffff !important;
        }
        @media (max-width: 720px) {
            .choice-inline-card-final {
                padding-right: 3.5rem;
            }
            .choice-inline-text-final {
                gap: 0.55rem;
                flex-wrap: wrap;
            }
            .choice-inline-title-text-final {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-inline-copy-final {
                font-size: 0.92rem;
                width: 100%;
            }
            .choice-inline-check-final .stButton {
                margin-left: -3.6rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-inline-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-inline-header-final">
            <div class="choice-inline-kicker-final">{kicker}</div>
            <div class="choice-inline-title-final">{title}</div>
            <div class="choice-inline-desc-final">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        text_col, check_col = st.columns([13.6, 0.7], gap="small")
        with text_col:
            st.markdown(
                f"""
                <div class="choice-inline-card-final{' is-selected' if is_active else ''}">
                    <div class="choice-inline-text-final">
                        <div class="choice-inline-title-text-final">{item['title']}</div>
                        <div class="choice-inline-copy-final{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with check_col:
            st.markdown('<div class="choice-inline-check-final">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_inline_selector_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_inline_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_inline_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-photo-wrap-final-fix2 {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-photo-header-final-fix2 {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-photo-kicker-final-fix2 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-photo-title-final-fix2 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-photo-desc-final-fix2 {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-photo-row-final-fix2 {
            margin-bottom: 0.5rem;
        }
        .choice-photo-card-final-fix2 {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #eadfd3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
        }
        .choice-photo-card-final-fix2.is-selected {
            border: 2px solid #5f4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #fffaf5;
        }
        .choice-photo-card-content-final-fix2 {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-radio-circle-fix2 {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 999px;
            border: 1.5px solid #d9cec3;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: transparent;
            margin-right: 0.5rem;
        }
        .choice-radio-circle-fix2.is-selected {
            background: #9b7a67;
            border-color: #9b7a67;
            color: #ffffff;
            font-weight: 800;
        }
        .choice-photo-card-title-final-fix2 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-photo-card-copy-final-fix2 {
            font-size: 1rem;
            line-height: 1.6;
            color: #b9ada5;
            flex: 1 1 auto;
        }
        .choice-photo-card-copy-final-fix2.is-selected {
            color: #8d7666;
        }
        .choice-photo-button-wrap-final-fix2 {
            margin-top: -72px;
            height: 72px;
            position: relative;
            z-index: 10;
        }
        .choice-photo-button-wrap-final-fix2 div[data-testid="stButton"] {
            height: 72px;
        }
        div[data-testid="stButton"] > button[kind="secondaryFormSubmit"],
        div[data-testid="stButton"] > button:not([kind="primary"]) {
            display: none;
        }
        .choice-photo-button-wrap-final-fix2 div[data-testid="stButton"] > button {
            width: 100%;
            height: 72px;
            opacity: 0;
            padding: 0;
            margin: 0;
            border: none;
            box-shadow: none;
            cursor: pointer;
        }
        @media (max-width: 760px) {
            .choice-photo-title-final-fix2 {
                font-size: 2.25rem;
            }
            .choice-photo-card-final-fix2 {
                padding: 0.95rem 1.2rem;
            }
            .choice-photo-card-content-final-fix2 {
                gap: 0.45rem;
                flex-wrap: wrap;
            }
            .choice-photo-card-title-final-fix2 {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-photo-card-copy-final-fix2 {
                width: 100%;
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-photo-wrap-final-fix2">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-photo-header-final-fix2">
            <div class="choice-photo-kicker-final-fix2">{kicker}</div>
            <div class="choice-photo-title-final-fix2">{title}</div>
            <div class="choice-photo-desc-final-fix2">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        selected_class = " is-selected" if is_active else ""
        check_char = "✓" if is_active else ""

        st.markdown(
            f"""
            <div class="choice-photo-row-final-fix2">
                <div class="choice-photo-card-final-fix2{selected_class}">
                    <div class="choice-photo-card-content-final-fix2">
                        <div class="choice-radio-circle-fix2{selected_class}">{check_char}</div>
                        <div class="choice-photo-card-title-final-fix2">{item["title"]}</div>
                        <div class="choice-photo-card-copy-final-fix2{selected_class}">{item["description"]}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="choice-photo-button-wrap-final-fix2">', unsafe_allow_html=True)
        if st.button(" ", key=f"{state_key}_pick_fix2_{index}", use_container_width=True):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_photo_fix2", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_photo_fix2",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-inline-wrap-final4 {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-inline-header-final4 {
            text-align: center;
            margin-bottom: 2.2rem;
        }
        .choice-inline-kicker-final4 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-inline-title-final4 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-inline-desc-final4 {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-inline-row-final4 {
            position: relative;
            margin-bottom: 1rem;
        }
        .choice-inline-card-final4 {
            background: rgba(255, 252, 248, 0.98);
            border: 1.3px solid #eadfd3;
            border-radius: 999px;
            min-height: 72px;
            display: flex;
            align-items: center;
            padding: 0.75rem 5rem 0.75rem 1.6rem;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            width: calc(100% + 4.7rem);
            margin-right: -4.7rem;
        }
        .choice-inline-card-final4.is-selected {
            border: 2px solid #2f2622;
            box-shadow: 0 10px 20px rgba(35, 31, 29, 0.08);
        }
        .choice-inline-text-final4 {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-inline-title-text-final4 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-inline-copy-final4 {
            font-size: 1rem;
            color: #b9ada5;
            line-height: 1.6;
            flex: 1 1 auto;
        }
        .choice-inline-copy-final4.is-selected {
            color: #9a7b67;
        }
        .choice-inline-check-final4 {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            min-height: 72px;
        }
        .choice-inline-check-final4 .stButton {
            display: flex !important;
            justify-content: flex-end !important;
            margin-left: -5rem !important;
            margin-top: -30px !important;
            z-index: 6;
        }
        .choice-inline-check-final4 .stButton > button {
            width: 42px !important;
            min-width: 42px !important;
            height: 42px !important;
            min-height: 42px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
            font-size: 1rem !important;
        }
        .choice-inline-check-final4 .stButton > button[kind="secondary"] {
            background: #ffffff !important;
            border: 1.5px solid #d8cec5 !important;
            color: transparent !important;
        }
        .choice-inline-check-final4 .stButton > button[kind="primary"] {
            background: #9b7a67 !important;
            border: 1.5px solid #9b7a67 !important;
            color: #ffffff !important;
        }
        @media (max-width: 720px) {
            .choice-inline-card-final4 {
                width: calc(100% + 3.7rem);
                margin-right: -3.7rem;
                padding-right: 4rem;
            }
            .choice-inline-text-final4 {
                gap: 0.55rem;
                flex-wrap: wrap;
            }
            .choice-inline-title-text-final4 {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-inline-copy-final4 {
                font-size: 0.92rem;
                width: 100%;
            }
            .choice-inline-check-final4 .stButton {
                margin-left: -4.1rem !important;
                margin-top: -30px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-inline-wrap-final4">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-inline-header-final4">
            <div class="choice-inline-kicker-final4">{kicker}</div>
            <div class="choice-inline-title-final4">{title}</div>
            <div class="choice-inline-desc-final4">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        text_col, check_col = st.columns([15.8, 0.45], gap="small")
        with text_col:
            st.markdown(
                f"""
                <div class="choice-inline-row-final4">
                    <div class="choice-inline-card-final4{' is-selected' if is_active else ''}">
                        <div class="choice-inline-text-final4">
                            <div class="choice-inline-title-text-final4">{item['title']}</div>
                            <div class="choice-inline-copy-final4{' is-selected' if is_active else ''}">{item['description']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with check_col:
            st.markdown('<div class="choice-inline-check-final4">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_inline_selector_final4_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_inline_final4", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_inline_final4",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-inline-wrap-final5 {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-inline-header-final5 {
            text-align: center;
            margin-bottom: 2.2rem;
        }
        .choice-inline-kicker-final5 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-inline-title-final5 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-inline-desc-final5 {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-inline-row-final5 {
            margin-bottom: 1rem;
        }
        .choice-inline-card-final5 {
            background: rgba(255, 252, 248, 0.98);
            border: 1.3px solid #eadfd3;
            border-radius: 999px;
            min-height: 72px;
            display: flex;
            align-items: center;
            padding: 0.75rem 4.9rem 0.75rem 1.6rem;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            width: calc(100% + 4.8rem);
            margin-right: -4.8rem;
        }
        .choice-inline-card-final5.is-selected {
            border: 2px solid #5f4637;
            box-shadow: 0 10px 20px rgba(35, 31, 29, 0.08);
            background: #fffaf5;
        }
        .choice-inline-text-final5 {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-inline-title-text-final5 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-inline-copy-final5 {
            font-size: 1rem;
            color: #b9ada5;
            line-height: 1.6;
            flex: 1 1 auto;
        }
        .choice-inline-copy-final5.is-selected {
            color: #8d7666;
        }
        .choice-inline-check-final5 {
            min-height: 72px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
        }
        .choice-inline-check-final5 .stButton {
            display: flex !important;
            justify-content: flex-end !important;
            margin-left: -5.05rem !important;
            z-index: 8;
        }
        .choice-inline-check-final5 .stButton > button {
            width: 44px !important;
            min-width: 44px !important;
            height: 44px !important;
            min-height: 44px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
        }
        .choice-inline-check-final5 .stButton > button[kind="secondary"] {
            background: #ffffff !important;
            border: 1.5px solid #d9cec3 !important;
            color: transparent !important;
        }
        .choice-inline-check-final5 .stButton > button[kind="primary"] {
            background: #9b7a67 !important;
            border: 1.5px solid #9b7a67 !important;
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            .choice-inline-card-final5 {
                width: calc(100% + 3.8rem);
                margin-right: -3.8rem;
                padding: 0.85rem 4rem 0.85rem 1.25rem;
            }
            .choice-inline-text-final5 {
                gap: 0.55rem;
                flex-wrap: wrap;
            }
            .choice-inline-title-text-final5 {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-inline-copy-final5 {
                font-size: 0.92rem;
                width: 100%;
            }
            .choice-inline-check-final5 .stButton {
                margin-left: -4.15rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-inline-wrap-final5">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-inline-header-final5">
            <div class="choice-inline-kicker-final5">{kicker}</div>
            <div class="choice-inline-title-final5">{title}</div>
            <div class="choice-inline-desc-final5">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        st.markdown('<div class="choice-inline-row-final5">', unsafe_allow_html=True)
        text_col, check_col = st.columns([16, 0.2], gap="small")
        with text_col:
            st.markdown(
                f"""
                <div class="choice-inline-card-final5{' is-selected' if is_active else ''}">
                    <div class="choice-inline-text-final5">
                        <div class="choice-inline-title-text-final5">{item['title']}</div>
                        <div class="choice-inline-copy-final5{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with check_col:
            st.markdown('<div class="choice-inline-check-final5">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_inline_selector_final5_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_inline_final5", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_inline_final5",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-photo-wrap-final {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-photo-header-final {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-photo-kicker-final {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-photo-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-photo-desc-final {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-photo-row-final {
            margin-bottom: 1.15rem;
        }
        .choice-photo-card-final {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #eadfd3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
        }
        .choice-photo-card-final.is-selected {
            border: 2px solid #5f4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #fffaf5;
        }
        .choice-photo-card-content-final {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-photo-card-title-final {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-photo-card-copy-final {
            font-size: 1rem;
            line-height: 1.6;
            color: #b9ada5;
            flex: 1 1 auto;
        }
        .choice-photo-card-copy-final.is-selected {
            color: #8d7666;
        }
        .choice-photo-circle-final {
            min-height: 72px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .choice-photo-circle-final .stButton {
            display: flex !important;
            justify-content: center !important;
        }
        .choice-photo-circle-final .stButton > button {
            width: 38px !important;
            min-width: 38px !important;
            height: 38px !important;
            min-height: 38px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
            font-size: 0.95rem !important;
            font-weight: 800 !important;
        }
        .choice-photo-circle-final .stButton > button[kind="secondary"] {
            background: #ffffff !important;
            border: 1.5px solid #d9cec3 !important;
            color: transparent !important;
        }
        .choice-photo-circle-final .stButton > button[kind="primary"] {
            background: #9b7a67 !important;
            border: 1.5px solid #9b7a67 !important;
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            .choice-photo-title-final {
                font-size: 2.25rem;
            }
            .choice-photo-card-final {
                padding: 0.95rem 1.2rem;
            }
            .choice-photo-card-content-final {
                gap: 0.45rem;
                flex-wrap: wrap;
            }
            .choice-photo-card-title-final {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-photo-card-copy-final {
                width: 100%;
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-photo-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-photo-header-final">
            <div class="choice-photo-kicker-final">{kicker}</div>
            <div class="choice-photo-title-final">{title}</div>
            <div class="choice-photo-desc-final">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        st.markdown('<div class="choice-photo-row-final">', unsafe_allow_html=True)
        circle_col, card_col = st.columns([0.85, 9.15], gap="small")
        with circle_col:
            st.markdown('<div class="choice-photo-circle-final">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_photo_pick_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with card_col:
            st.markdown(
                f"""
                <div class="choice-photo-card-final{' is-selected' if is_active else ''}">
                    <div class="choice-photo-card-content-final">
                        <div class="choice-photo-card-title-final">{item['title']}</div>
                        <div class="choice-photo-card-copy-final{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_photo_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_photo_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-wrap {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-header {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-kicker {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-title {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-desc {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-card {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #EADFD3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }
        .choice-card.is-selected {
            border: 2px solid #5F4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #FFFAF5;
        }
        .choice-card-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
        }
        .choice-card-desc {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
        }
        .choice-card-desc.is-selected {
            color: #8D7666;
        }
        .radio-col {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 72px;
            margin-bottom: 0.5rem;
        }
        .radio-col .stButton {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .radio-col .stButton > button {
            width: 44px !important;
            min-width: 44px !important;
            height: 44px !important;
            min-height: 44px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .radio-col .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 2px solid #D9CEC3 !important;
            color: transparent !important;
        }
        .radio-col .stButton > button[kind="primary"] {
            background: #9B7A67 !important;
            border: 2px solid #9B7A67 !important;
            color: #FFFFFF !important;
        }
        @media (max-width: 760px) {
            .choice-title { font-size: 2.25rem; }
            .choice-card { padding: 0.95rem 1.2rem; }
            .choice-card-title { font-size: 1.35rem; }
            .choice-card-desc { font-size: 0.92rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-wrap">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-header">
            <div class="choice-kicker">{kicker}</div>
            <div class="choice-title">{title}</div>
            <div class="choice-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)

    for index, item in enumerate(options):
        is_active = selected == item["key"]
        radio_col, card_col = st.columns([1, 9], gap="small")

        with radio_col:
            st.markdown('<div class="radio-col">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_pick_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with card_col:
            st.markdown(
                f"""
                <div class="choice-card{' is-selected' if is_active else ''}">
                    <div class="choice-card-title">{item['title']}</div>
                    <div class="choice-card-desc{' is-selected' if is_active else ''}">{item['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_native_featured_product_card(personal_color: str, featured_product: Dict[str, str]) -> None:
    theme = PERSONAL_COLOR_THEME.get(personal_color, {"accent": "#a97757", "button": "#a97757"})
    image_markup = build_product_image_markup(
        featured_product["name"],
        "featured-product-image featured-product-image-large",
    )
    skin_profile = build_skin_profile_phrase()
    skin_recommendation_line = build_skin_profile_recommendation_line()
    st.markdown(
        """
        <style>
        .result-featured-product-card.product-large {
            max-width: 680px !important;
            padding: 1.2rem 1.25rem 1.2rem 1.25rem !important;
        }
        .featured-recommend-top {
            text-align: center;
            margin: 0.2rem 0 1rem 0;
        }
        .featured-recommend-title {
            font-size: 1.32rem;
            font-weight: 800;
            color: #2f2622;
            margin-bottom: 0.45rem;
        }
        .featured-recommend-copy {
            font-size: 1rem;
            line-height: 1.8;
            color: #4a403a;
            font-weight: 700;
        }
        .featured-product-image-large {
            width: 100% !important;
            max-height: none !important;
            min-height: 620px !important;
            object-fit: contain !important;
            display: block !important;
            margin: 0 auto 0.9rem auto !important;
            background: #ffffff !important;
            border-radius: 22px !important;
        }
        @media (max-width: 760px) {
            .result-featured-product-card.product-large {
                max-width: 94vw !important;
            }
            .featured-product-image-large {
                min-height: 520px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    top_recommend_markup = (
        f'<div class="featured-recommend-top">'
        f'<div class="featured-recommend-title">{skin_profile}</div>'
        f'<div class="featured-recommend-copy">{skin_recommendation_line}</div>'
        f'</div>'
    )

    body_parts = [top_recommend_markup, image_markup]
    if not image_markup:
        body_parts.extend(
            [
                f'<div style="font-size:1.42rem; font-weight:800; color:#2f2622; margin:0.95rem 0 0.55rem 0;">{featured_product["name"]}</div>',
                f'<div class="featured-product-note">{featured_product["notes"]}</div>',
            ]
        )
    featured_body = "".join([part for part in body_parts if part])

    st.markdown(
        f"""
        <div class="result-featured-product-card product-large" style="border-color:{hex_to_rgba(theme['accent'], 0.24)};">
            <div style="font-size:0.92rem; font-weight:700; color:{theme['accent']}; margin-bottom:0.55rem;">유쏘풀 추천 향수</div>
            {featured_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_integrated_feedback_section() -> None:
    ensure_integrated_feedback_state()
    predicted_label = st.session_state.personal_color
    class_options = list(load_personal_color_assets()["classes"])
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stTextArea"] label p {
            color:#8a776a !important;
            font-weight:700 !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            background: rgba(255,251,247,0.96) !important;
            border: 1.4px solid #dcc7b6 !important;
            border-radius: 20px !important;
            color:#2f2622 !important;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 132px !important;
            box-shadow: 0 14px 28px rgba(199,171,145,0.10) !important;
            padding: 1rem 1.05rem !important;
            line-height: 1.7 !important;
            -webkit-text-fill-color:#2f2622 !important;
            caret-color:#2f2622 !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color:#2f2622 !important;
            -webkit-text-fill-color:#2f2622 !important;
            opacity:1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    body_left, body_center, body_right = st.columns([0.45, 5.2, 0.45])
    with body_center:
        st.markdown("##### 예측 결과가 맞나요?")
        st.markdown(f"**{predicted_label}**")
        render_feedback_choice_buttons("feedback_match_answer", ["맞아요", "아니에요"], "feedback_match_final")
        is_match = st.session_state.feedback_match_answer == "맞아요"
        actual_label = predicted_label

        if not is_match:
            st.markdown("##### 실제 퍼스널 컬러를 알려주세요")
            actual_label = st.selectbox(
                "실제 퍼스널 컬러",
                class_options,
                index=class_options.index(predicted_label) if predicted_label in class_options else 0,
                key="feedback_actual_label_final",
                label_visibility="collapsed",
            )

        st.markdown("##### 이 이미지를 학습 데이터로 사용해도 될까요?")
        render_feedback_choice_buttons("feedback_consent_answer", ["동의", "비동의"], "feedback_consent_final")
        consent = st.session_state.feedback_consent_answer == "동의"

        st.markdown("##### 어떤 점이 부족했는지, 추가됐으면 하는 점이 있다면 적어주세요.")
        st.text_area(
            "추가 피드백",
            key="feedback_comment",
            placeholder="예: 설명이 더 자세했으면 좋겠어요. 추천 향 이유를 더 보고 싶어요.",
            height=132,
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        _, submit_col, _ = st.columns([1, 1.6, 1])
        with submit_col:
            if st.button("피드백 저장", key="save_integrated_feedback_final", type="primary", use_container_width=True):
                submit_integrated_feedback(actual_label=actual_label, is_match=is_match, consent=consent)

        if st.session_state.feedback_saved and st.session_state.feedback_saved_message:
            st.success(st.session_state.feedback_saved_message)


def render_product_feedback_page() -> None:
    render_step_progress()
    run_recommendation()
    selected_color = st.session_state.personal_color
    products = st.session_state.recommended_products or PRODUCT_LIBRARY[st.session_state.recommended_label]
    recommended_labels = st.session_state.recommended_labels

    featured_product = pick_featured_product(products)
    if featured_product:
        render_native_featured_product_card(selected_color, featured_product)

    render_integrated_feedback_section()

    st.markdown("<div style='height:2.1rem;'></div>", unsafe_allow_html=True)
    left_btn, center_btn, right_btn = st.columns(3)
    with left_btn:
        if st.button("이전", key="back_to_scent_result_final", use_container_width=True):
            go_to("scent_result")
    with center_btn:
        if st.button("사진 다시 선택", key="back_to_face_final", use_container_width=True):
            go_to("face")
    with right_btn:
        if st.button("처음부터 다시", key="restart_product_feedback_final", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-photo-wrap-final {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-photo-header-final {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-photo-kicker-final {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-photo-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-photo-desc-final {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-photo-row-final {
            margin-bottom: 0.5rem;
            position: relative;
        }
        .choice-photo-card-final {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #EADFD3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
        }
        .choice-photo-card-final.is-selected {
            border: 2px solid #5F4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #FFFAF5;
        }
        .choice-photo-card-content-final {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-photo-card-title-final {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-photo-card-copy-final {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
            flex: 1 1 auto;
        }
        .choice-photo-card-copy-final.is-selected {
            color: #8D7666;
        }
        .choice-radio-circle {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 999px;
            border: 1.5px solid #D9CEC3;
            background: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: transparent;
            margin-right: 0.5rem;
        }
        .choice-radio-circle.is-selected {
            background: #9B7A67;
            border-color: #9B7A67;
            color: #FFFFFF;
            font-weight: 800;
        }
        .choice-photo-row-final div[data-testid="stButton"] > button {
            position: relative;
            margin-top: -60px;
            height: 60px;
            opacity: 0;
            z-index: 10;
            cursor: pointer;
            width: 100%;
        }
        @media (max-width: 760px) {
            .choice-photo-title-final {
                font-size: 2.25rem;
            }
            .choice-photo-card-final {
                padding: 0.95rem 1.2rem;
            }
            .choice-photo-card-content-final {
                gap: 0.45rem;
                flex-wrap: wrap;
            }
            .choice-photo-card-title-final {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-photo-card-copy-final {
                width: 100%;
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-photo-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-photo-header-final">
            <div class="choice-photo-kicker-final">{kicker}</div>
            <div class="choice-photo-title-final">{title}</div>
            <div class="choice-photo-desc-final">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)

    for index, item in enumerate(options):
        is_active = selected == item["key"]

        st.markdown(
            f"""
            <div class="choice-photo-row-final">
                <div class="choice-photo-card-final{' is-selected' if is_active else ''}">
                    <div class="choice-photo-card-content-final">
                        <div class="choice-radio-circle{' is-selected' if is_active else ''}">
                            {'✓' if is_active else ''}
                        </div>
                        <div class="choice-photo-card-title-final">{item['title']}</div>
                        <div class="choice-photo-card-copy-final{' is-selected' if is_active else ''}">{item['description']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(" ", key=f"{state_key}_pick_{index}", use_container_width=True):
            st.session_state[state_key] = item["key"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_photo_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_photo_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-photo-wrap-final-fix {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-photo-header-final-fix {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-photo-kicker-final-fix {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.9rem;
        }
        .choice-photo-title-final-fix {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2f2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-photo-desc-final-fix {
            color: #7d7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-photo-row-final-fix {
            margin-bottom: 0.5rem;
            position: relative;
        }
        .choice-photo-card-final-fix {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #eadfd3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
        }
        .choice-photo-card-final-fix.is-selected {
            border: 2px solid #5f4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #fffaf5;
        }
        .choice-photo-card-content-final-fix {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-photo-card-title-final-fix {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8e5f41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-photo-card-copy-final-fix {
            font-size: 1rem;
            line-height: 1.6;
            color: #b9ada5;
            flex: 1 1 auto;
        }
        .choice-photo-card-copy-final-fix.is-selected {
            color: #8d7666;
        }
        .choice-radio-circle-fix {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 999px;
            border: 1.5px solid #d9cec3;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: transparent;
            margin-right: 0.5rem;
        }
        .choice-radio-circle-fix.is-selected {
            background: #9b7a67;
            border-color: #9b7a67;
            color: #ffffff;
            font-weight: 800;
        }
        .choice-photo-row-final-fix + div[data-testid="stButton"],
        .choice-photo-row-final-fix ~ div[data-testid="stButton"] {
            display: none;
        }
        @media (max-width: 760px) {
            .choice-photo-title-final-fix {
                font-size: 2.25rem;
            }
            .choice-photo-card-final-fix {
                padding: 0.95rem 1.2rem;
            }
            .choice-photo-card-content-final-fix {
                gap: 0.45rem;
                flex-wrap: wrap;
            }
            .choice-photo-card-title-final-fix {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-photo-card-copy-final-fix {
                width: 100%;
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-photo-wrap-final-fix">', unsafe_allow_html=True)
    st.markdown(
        (
            f'<div class="choice-photo-header-final-fix">'
            f'<div class="choice-photo-kicker-final-fix">{kicker}</div>'
            f'<div class="choice-photo-title-final-fix">{title}</div>'
            f'<div class="choice-photo-desc-final-fix">{description}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        selected_class = " is-selected" if is_active else ""
        check_char = "✓" if is_active else ""
        row_markup = (
            f'<div class="choice-photo-row-final-fix">'
            f'<div class="choice-photo-card-final-fix{selected_class}">'
            f'<div class="choice-photo-card-content-final-fix">'
            f'<div class="choice-radio-circle-fix{selected_class}">{check_char}</div>'
            f'<div class="choice-photo-card-title-final-fix">{item["title"]}</div>'
            f'<div class="choice-photo-card-copy-final-fix{selected_class}">{item["description"]}</div>'
            f'</div></div></div>'
        )
        st.markdown(row_markup, unsafe_allow_html=True)

        if st.button(" ", key=f"{state_key}_pick_fix_{index}", use_container_width=True):
            st.session_state[state_key] = item["key"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_photo_fix", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_photo_fix",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-photo-wrap-final {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-photo-header-final {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-photo-kicker-final {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-photo-title-final {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-photo-desc-final {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-photo-row-final {
            margin-bottom: 0.5rem;
            position: relative;
        }
        .choice-photo-card-final {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #EADFD3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
        }
        .choice-photo-card-final.is-selected {
            border: 2px solid #5F4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #FFFAF5;
        }
        .choice-photo-card-content-final {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }
        .choice-photo-card-title-final {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-photo-card-copy-final {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
            flex: 1 1 auto;
        }
        .choice-photo-card-copy-final.is-selected {
            color: #8D7666;
        }
        .choice-radio-circle {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 999px;
            border: 1.5px solid #D9CEC3;
            background: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: transparent;
            margin-right: 0.5rem;
        }
        .choice-radio-circle.is-selected {
            background: #9B7A67;
            border-color: #9B7A67;
            color: #FFFFFF;
            font-weight: 800;
        }
        @media (max-width: 760px) {
            .choice-photo-title-final {
                font-size: 2.25rem;
            }
            .choice-photo-card-final {
                padding: 0.95rem 1.2rem;
            }
            .choice-photo-card-content-final {
                gap: 0.45rem;
                flex-wrap: wrap;
            }
            .choice-photo-card-title-final {
                font-size: 1.35rem;
                width: 100%;
            }
            .choice-photo-card-copy-final {
                width: 100%;
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-photo-wrap-final">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-photo-header-final">
            <div class="choice-photo-kicker-final">{kicker}</div>
            <div class="choice-photo-title-final">{title}</div>
            <div class="choice-photo-desc-final">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)

    for index, item in enumerate(options):
        is_active = selected == item["key"]

        btn_style = f"""
        <style>
        div[data-testid="stButton"]:has(button[key="{state_key}_pick_{index}"]) > button {{
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: {"#FFFAF5" if is_active else "rgba(255, 253, 249, 0.98)"} !important;
            border: {"2px solid #5F4637" if is_active else "1.25px solid #EADFD3"} !important;
            box-shadow: {"0 10px 22px rgba(73,53,40,0.10)" if is_active else "0 8px 18px rgba(199,171,145,0.06)"} !important;
            width: 100% !important;
            text-align: left !important;
            color: #2F2622 !important;
            margin-bottom: 0.5rem;
        }}
        </style>
        """
        st.markdown(btn_style, unsafe_allow_html=True)

        label = f"{'✓' if is_active else '○'}  **{item['title']}**  {item['description']}"

        if st.button(label, key=f"{state_key}_pick_{index}", use_container_width=True):
            st.session_state[state_key] = item["key"]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_photo_final", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_photo_final",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-wrap-final-live {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-header-final-live {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-kicker-final-live {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-title-final-live {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-desc-final-live {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-card-final-live {
            min-height: 72px;
            border-radius: 24px;
            padding: 1.05rem 1.8rem;
            background: rgba(255, 253, 249, 0.98);
            border: 1.25px solid #EADFD3;
            box-shadow: 0 8px 18px rgba(199, 171, 145, 0.06);
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }
        .choice-card-final-live.is-selected {
            border: 2px solid #5F4637;
            box-shadow: 0 10px 22px rgba(73, 53, 40, 0.10);
            background: #FFFAF5;
        }
        .choice-card-title-final-live {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
        }
        .choice-card-desc-final-live {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
        }
        .choice-card-desc-final-live.is-selected {
            color: #8D7666;
        }
        .radio-col-final-live {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            min-height: 72px;
            margin-bottom: 0.5rem;
            padding-right: 0.15rem;
        }
        .radio-col-final-live .stButton {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .radio-col-final-live .stButton > button {
            width: 44px !important;
            min-width: 44px !important;
            height: 44px !important;
            min-height: 44px !important;
            border-radius: 999px !important;
            padding: 0 !important;
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .radio-col-final-live .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 2px solid #D9CEC3 !important;
            color: transparent !important;
        }
        .radio-col-final-live .stButton > button[kind="primary"] {
            background: #9B7A67 !important;
            border: 2px solid #9B7A67 !important;
            color: #FFFFFF !important;
        }
        @media (max-width: 760px) {
            .choice-title-final-live { font-size: 2.25rem; }
            .choice-card-final-live { padding: 0.95rem 1.2rem; }
            .choice-card-title-final-live { font-size: 1.35rem; }
            .choice-card-desc-final-live { font-size: 0.92rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-wrap-final-live">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-header-final-live">
            <div class="choice-kicker-final-live">{kicker}</div>
            <div class="choice-title-final-live">{title}</div>
            <div class="choice-desc-final-live">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)

    for index, item in enumerate(options):
        is_active = selected == item["key"]
        radio_col, card_col = st.columns([0.42, 9.58], gap="small")

        with radio_col:
            st.markdown('<div class="radio-col-final-live">', unsafe_allow_html=True)
            if st.button(
                "✓" if is_active else " ",
                key=f"{state_key}_pick_final_live_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=False,
            ):
                st.session_state[state_key] = item["key"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with card_col:
            st.markdown(
                f"""
                <div class="choice-card-final-live{' is-selected' if is_active else ''}">
                    <div class="choice-card-title-final-live">{item['title']}</div>
                    <div class="choice-card-desc-final-live{' is-selected' if is_active else ''}">{item['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_final_live", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_final_live",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-clean-wrap-v2 {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-clean-header-v2 {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-clean-kicker-v2 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-clean-title-v2 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-clean-desc-v2 {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-clean-info-card-v2 {
            min-height: 60px;
            border-radius: 24px;
            padding: 0.95rem 1.35rem;
            background: #fffdfa;
            border: 1.25px solid #eadfd3;
            box-shadow: 0 8px 18px rgba(199,171,145,0.06);
            width: 100%;
            margin-bottom: 0.45rem;
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .choice-clean-info-circle-v2 {
            width: 26px;
            height: 26px;
            min-width: 26px;
            border-radius: 999px;
            border: 1.5px solid #d9cec3;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            color: transparent;
            font-size: 0.92rem;
            font-weight: 800;
        }
        .choice-clean-info-circle-v2.is-selected {
            background: #9B7A67;
            border-color: #9B7A67;
            color: #ffffff;
        }
        .choice-clean-info-title-v2 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-clean-info-copy-v2 {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
            flex: 1 1 auto;
        }
        .choice-clean-info-copy-v2.is-selected {
            color: #8D7666;
        }
        .choice-clean-select-v2 .stButton > button {
            min-height: 52px !important;
            border-radius: 24px !important;
            width: 100% !important;
            border: 1.25px solid #e1c5ab !important;
            background: #fffdfa !important;
            color: #8E5F41 !important;
            font-weight: 700 !important;
            margin-bottom: 0.85rem !important;
        }
        .choice-clean-select-v2 .stButton > button[kind="primary"] {
            background: #9B7A67 !important;
            border: 1.25px solid #9B7A67 !important;
            color: #FFFFFF !important;
        }
        @media (max-width: 760px) {
            .choice-clean-title-v2 { font-size: 2.25rem; }
            .choice-clean-info-card-v2 {
                padding: 0.9rem 1.1rem;
            }
            .choice-clean-info-title-v2 {
                font-size: 1.35rem;
            }
            .choice-clean-info-copy-v2 {
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-clean-wrap-v2">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-clean-header-v2">
            <div class="choice-clean-kicker-v2">{kicker}</div>
            <div class="choice-clean-title-v2">{title}</div>
            <div class="choice-clean-desc-v2">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        select_label = f"{item['title']} 선택"

        st.markdown(
            f"""
            <div class="choice-clean-info-card-v2">
                <div class="choice-clean-info-circle-v2{' is-selected' if is_active else ''}">{'✓' if is_active else ''}</div>
                <div class="choice-clean-info-title-v2">{item['title']}</div>
                <div class="choice-clean-info-copy-v2{' is-selected' if is_active else ''}">{item['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="choice-clean-select-v2">', unsafe_allow_html=True)
        if st.button(
            select_label,
            key=f"{state_key}_select_clean_v2_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_clean_last_v2", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_clean_last_v2",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)


def render_skin_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_choice_page(kicker, title, description, options, state_key, prev_page, next_page)


def render_choice_page(
    kicker: str,
    title: str,
    description: str,
    options: List[Dict[str, object]],
    state_key: str,
    prev_page: str,
    next_page: str,
) -> None:
    render_step_progress()
    st.markdown(
        """
        <style>
        .choice-clean-wrap-v4 {
            max-width: 840px;
            margin: 0 auto;
        }
        .choice-clean-header-v4 {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .choice-clean-kicker-v4 {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #A86A3C;
            margin-bottom: 0.9rem;
        }
        .choice-clean-title-v4 {
            font-size: 3rem;
            line-height: 1.2;
            font-weight: 900;
            color: #2F2622;
            letter-spacing: -0.04em;
            white-space: pre-line;
            margin-bottom: 0.95rem;
        }
        .choice-clean-desc-v4 {
            color: #7D7068;
            font-size: 1.08rem;
            line-height: 1.82;
            white-space: pre-line;
        }
        .choice-clean-info-card-v4 {
            min-height: 60px;
            border-radius: 24px;
            padding: 0.95rem 1.35rem;
            background: #fffdfa;
            border: 1.25px solid #eadfd3;
            box-shadow: 0 8px 18px rgba(199,171,145,0.06);
            width: 100%;
            margin-bottom: 0.45rem;
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .choice-clean-info-circle-v4 {
            width: 26px;
            height: 26px;
            min-width: 26px;
            border-radius: 999px;
            border: 1.5px solid #d9cec3;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            color: transparent;
            font-size: 0.92rem;
            font-weight: 800;
        }
        .choice-clean-info-circle-v4.is-selected {
            background: #9B7A67;
            border-color: #9B7A67;
            color: #ffffff;
        }
        .choice-clean-info-title-v4 {
            font-size: 1.7rem;
            font-weight: 800;
            color: #8E5F41;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .choice-clean-info-copy-v4 {
            font-size: 1rem;
            line-height: 1.6;
            color: #B9ADA5;
            flex: 1 1 auto;
        }
        .choice-clean-info-copy-v4.is-selected {
            color: #8D7666;
        }
        .choice-clean-select-v4 .stButton > button {
            min-height: 52px !important;
            border-radius: 24px !important;
            width: 100% !important;
            border: 1.25px solid #e1c5ab !important;
            background: #fffdfa !important;
            color: #8E5F41 !important;
            font-weight: 700 !important;
            margin-bottom: 0.85rem !important;
        }
        .choice-clean-select-v4 .stButton > button[kind="primary"] {
            background: #9B7A67 !important;
            border: 1.25px solid #9B7A67 !important;
            color: #FFFFFF !important;
        }
        @media (max-width: 760px) {
            .choice-clean-title-v4 { font-size: 2.25rem; }
            .choice-clean-info-card-v4 { padding: 0.9rem 1.1rem; }
            .choice-clean-info-title-v4 { font-size: 1.35rem; }
            .choice-clean-info-copy-v4 { font-size: 0.92rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="choice-clean-wrap-v4">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="choice-clean-header-v4">
            <div class="choice-clean-kicker-v4">{kicker}</div>
            <div class="choice-clean-title-v4">{title}</div>
            <div class="choice-clean-desc-v4">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.get(state_key)
    for index, item in enumerate(options):
        is_active = selected == item["key"]
        select_label = f"{item['title']} 선택"
        st.markdown(
            f"""
            <div class="choice-clean-info-card-v4">
                <div class="choice-clean-info-circle-v4{' is-selected' if is_active else ''}">{'✓' if is_active else ''}</div>
                <div class="choice-clean-info-title-v4">{item['title']}</div>
                <div class="choice-clean-info-copy-v4{' is-selected' if is_active else ''}">{item['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="choice-clean-select-v4">', unsafe_allow_html=True)
        if st.button(
            select_label,
            key=f"{state_key}_select_clean_v4_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state[state_key] = item["key"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        if st.button("이전", key=f"prev_{state_key}_clean_last_v4", use_container_width=True):
            go_to(prev_page)
    with right:
        if st.button(
            "다음",
            key=f"next_{state_key}_clean_last_v4",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get(state_key)),
        ):
            go_to(next_page)

def save_integrated_feedback_log(
    *,
    uploaded_name: str,
    predicted_label: str,
    predicted_confidence: Optional[float],
    user_label: str,
    is_match: bool,
    consent: bool,
    feedback_comment: str = "",
    saved_image_path: str = "",
) -> str:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "uploaded_name": uploaded_name,
        "predicted_label": predicted_label,
        "predicted_confidence": f"{predicted_confidence:.4f}" if predicted_confidence is not None else "",
        "user_label": user_label,
        "is_match": "yes" if is_match else "no",
        "consent": "yes" if consent else "no",
        "feedback_comment": feedback_comment,
        "saved_image_path": saved_image_path,
    }

    sheet_destination = append_feedback_to_google_sheet(row)
    if sheet_destination:
        return sheet_destination

    ensure_feedback_dirs()
    is_new = not FEEDBACK_LOG_PATH.exists()
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "uploaded_name",
                "predicted_label",
                "predicted_confidence",
                "user_label",
                "is_match",
                "consent",
                "feedback_comment",
                "saved_image_path",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return str(FEEDBACK_LOG_PATH)


def submit_integrated_feedback(actual_label: str, is_match: bool, consent: bool) -> None:
    uploaded_file = st.session_state.uploaded_image
    if uploaded_file is None:
        return

    saved_image_path = ""
    if consent:
        saved_image_path = save_integrated_feedback_image(uploaded_file, actual_label)

    feedback_log_destination = save_integrated_feedback_log(
        uploaded_name=getattr(uploaded_file, "name", "camera_capture"),
        predicted_label=st.session_state.personal_color,
        predicted_confidence=st.session_state.prediction_confidence,
        user_label=actual_label,
        is_match=is_match,
        consent=consent,
        feedback_comment=st.session_state.get("feedback_comment", "").strip(),
        saved_image_path=saved_image_path,
    )

    st.session_state.feedback_saved = True
    if consent:
        st.session_state.feedback_saved_message = (
            "피드백과 이미지 사용 동의가 저장되었어요.\n"
            f"피드백 저장 위치: {feedback_log_destination}\n"
            f"동의 이미지 저장 위치: {saved_image_path}"
        )
    else:
        st.session_state.feedback_saved_message = (
            "피드백이 저장되었고 이미지는 학습 데이터로 사용하지 않도록 기록했어요.\n"
            f"피드백 저장 위치: {feedback_log_destination}"
        )


GOOGLE_FEEDBACK_FORM_URL = "https://forms.gle/8xHNzEhk3mt3EJya8"


def render_integrated_feedback_section() -> None:
    st.markdown(
        """
        <style>
        .feedback-form-wrap {
            max-width: 860px;
            margin: 0 auto;
        }
        .feedback-form-card {
            background: linear-gradient(180deg, rgba(255,251,247,0.98) 0%, rgba(255,248,242,0.98) 100%);
            border: 1.35px solid #dcc7b6;
            border-radius: 28px;
            padding: 1.75rem 1.6rem 1.6rem 1.6rem;
            box-shadow: 0 16px 34px rgba(199,171,145,0.12);
        }
        .feedback-form-kicker {
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #a86a3c;
            margin-bottom: 0.8rem;
        }
        .feedback-form-title {
            font-size: 1.85rem;
            line-height: 1.35;
            font-weight: 900;
            color: #2f2622;
            margin-bottom: 0.95rem;
            white-space: pre-line;
        }
        .feedback-form-copy {
            color: #7d7068;
            font-size: 1rem;
            line-height: 1.8;
            margin-bottom: 1rem;
        }
        .feedback-form-alert {
            background: rgba(255,244,236,0.92);
            border: 1.2px solid #e5c5ad;
            border-radius: 22px;
            padding: 1rem 1.05rem;
            color: #7a4a2d;
            font-size: 1rem;
            line-height: 1.7;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .feedback-form-note {
            color: #7d7068;
            font-size: 0.98rem;
            line-height: 1.8;
            margin-bottom: 1.3rem;
        }
        .feedback-form-link-note {
            margin-top: 0.9rem;
            color: #9a887c;
            font-size: 0.9rem;
            line-height: 1.6;
            word-break: break-all;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="feedback-form-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="feedback-form-card">
            <div class="feedback-form-kicker">FEEDBACK</div>
            <div class="feedback-form-title">추가 피드백은 구글 폼으로 남겨주세요.</div>
            <div class="feedback-form-copy">
                결과를 사용해보시며 느낀 점이나 개선되었으면 하는 부분이 있다면 구글 폼으로 의견을 남겨주세요.
                남겨주신 피드백은 서비스 완성도를 높이는 데 큰 도움이 됩니다.
            </div>
            <div class="feedback-form-alert">
                퍼스널 컬러 분석 개선을 위해 얼굴 이미지(셀카)를 제공해주세요 !<br/>
                :rotating_light: 제공해주신 이미지는 머신러닝 모델 학습용으로만 사용됩니다.
            </div>
            <div class="feedback-form-note">
                분석 결과가 얼마나 잘 맞았는지, 어떤 점이 아쉬웠는지, 추가되면 좋을 기능이 무엇인지 자유롭게 적어주세요.
                구글 폼에서 이미지를 함께 제출해주시면 이후 모델 개선에도 반영할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("구글 폼으로 피드백 남기기 →", GOOGLE_FEEDBACK_FORM_URL, use_container_width=True)
    st.markdown(
        f'<div class="feedback-form-link-note">폼 링크: {GOOGLE_FEEDBACK_FORM_URL}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
