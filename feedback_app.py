# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

import streamlit as st

import deok


APP_DIR = Path(__file__).resolve().parent
FEEDBACK_DIR = APP_DIR / "feedback_data"
CONSENTED_IMAGE_DIR = FEEDBACK_DIR / "consented_images"
FEEDBACK_LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"


def ensure_feedback_dirs() -> None:
    FEEDBACK_DIR.mkdir(exist_ok=True)
    CONSENTED_IMAGE_DIR.mkdir(exist_ok=True)


def init_feedback_session() -> None:
    defaults = {
        "feedback_prediction": None,
        "feedback_uploaded_name": None,
        "feedback_saved": False,
        "feedback_saved_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_feedback_log(
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


def save_consented_image(uploaded_file, user_label: str) -> str:
    ensure_feedback_dirs()
    original_suffix = Path(uploaded_file.name).suffix or ".png"
    safe_label = deok.normalize_asset_key(user_label) or "unknown"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}_{uuid.uuid4().hex[:8]}{original_suffix}"
    target_path = CONSENTED_IMAGE_DIR / filename
    target_path.write_bytes(uploaded_file.getvalue())
    return str(target_path)


def run_prediction(uploaded_file) -> None:
    prediction = deok.predict_personal_color_from_image(uploaded_file)
    st.session_state.feedback_prediction = prediction
    st.session_state.feedback_uploaded_name = uploaded_file.name
    st.session_state.feedback_saved = False
    st.session_state.feedback_saved_message = None


def submit_feedback(uploaded_file, actual_label: str, is_match: bool, consent: bool) -> None:
    prediction = st.session_state.feedback_prediction
    if not prediction:
        return

    saved_image_path = ""
    if consent:
        saved_image_path = save_consented_image(uploaded_file, actual_label)

    save_feedback_log(
        uploaded_name=st.session_state.feedback_uploaded_name or uploaded_file.name,
        predicted_label=prediction["label"],
        predicted_confidence=prediction.get("confidence"),
        user_label=actual_label,
        is_match=is_match,
        consent=consent,
        saved_image_path=saved_image_path,
    )

    st.session_state.feedback_saved = True
    if consent:
        st.session_state.feedback_saved_message = "동의해주셔서 감사합니다. 이미지와 정답 라벨을 학습용 후보 데이터로 저장했어요."
    else:
        st.session_state.feedback_saved_message = "응답을 기록했어요. 이미지는 저장하지 않았습니다."


def render_feedback_questions(uploaded_file) -> None:
    prediction = st.session_state.feedback_prediction
    if not prediction:
        return

    predicted_label = prediction["label"]
    confidence = prediction.get("confidence")
    class_options = list(prediction.get("classes") or [])

    st.markdown("### 결과 확인")
    st.markdown(f"1. 예측 결과가 `{predicted_label}`인데 맞나요?")
    if confidence is not None:
        st.caption(f"예측 신뢰도: {confidence:.1%}")

    match_answer = st.radio(
        "결과 확인",
        ["맞아요", "아니에요"],
        horizontal=True,
        label_visibility="collapsed",
        key="feedback_match_answer",
    )

    actual_label = predicted_label
    is_match = match_answer == "맞아요"

    if not is_match:
        st.markdown("2. 실제 퍼스널 컬러를 알려주세요.")
        if class_options:
            default_index = class_options.index(predicted_label) if predicted_label in class_options else 0
            actual_label = st.selectbox(
                "실제 퍼스널 컬러",
                class_options,
                index=default_index,
                key="feedback_actual_label",
            )
        else:
            actual_label = st.text_input("실제 퍼스널 컬러", key="feedback_actual_label_text").strip() or predicted_label

    consent_question = (
        "3. 결과가 맞으니 이 이미지와 라벨을 정답 데이터로 학습에 사용해도 될까요?"
        if is_match
        else "3. 모델 개선을 위해 이 이미지와 수정한 정답 라벨을 학습 데이터로 사용해도 될까요?"
    )
    st.markdown(consent_question)

    consent_answer = st.radio(
        "학습 동의",
        ["동의", "비동의"],
        horizontal=True,
        label_visibility="collapsed",
        key="feedback_consent_answer",
    )

    left, right = st.columns([1, 1])
    with left:
        if st.button("피드백 저장", type="primary", use_container_width=True, key="feedback_submit"):
            submit_feedback(
                uploaded_file=uploaded_file,
                actual_label=actual_label,
                is_match=is_match,
                consent=consent_answer == "동의",
            )
    with right:
        if st.button("다시 분석", use_container_width=True, key="feedback_reset"):
            st.session_state.feedback_prediction = None
            st.session_state.feedback_uploaded_name = None
            st.session_state.feedback_saved = False
            st.session_state.feedback_saved_message = None
            st.rerun()

    if st.session_state.feedback_saved_message:
        st.success(st.session_state.feedback_saved_message)


def main() -> None:
    init_feedback_session()

    st.title("퍼스널 컬러 피드백")
    st.write(
        "퍼스널 컬러를 이미 알고 있는 사용자가 결과를 검증하고, "
        "원할 경우 모델 개선을 위한 학습 데이터 사용에 동의할 수 있는 페이지입니다."
    )

    uploaded_file = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    st.image(uploaded_file, caption="업로드한 이미지", width=320)

    if st.session_state.feedback_prediction is None or st.session_state.feedback_uploaded_name != uploaded_file.name:
        if st.button("퍼스널 컬러 분석", type="primary", use_container_width=True):
            with st.spinner("퍼스널 컬러를 분석 중입니다..."):
                run_prediction(uploaded_file)
            st.rerun()
        return

    prediction = st.session_state.feedback_prediction
    st.info(f"예측 결과: {prediction['label']}")
    render_feedback_questions(uploaded_file)


if __name__ == "__main__":
    main()
