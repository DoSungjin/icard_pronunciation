# Conda 환경 관리 가이드

생성 · 복제 · 내보내기 · ipykernel 연결 · 정리까지 한 번에 정리한 실전 치트시트.
예시는 icard 프로젝트(발음 평가 앱) 환경 기준으로 작성했습니다.

---

## 목차
1. [개념 한 줄 정리](#1-개념-한-줄-정리)
2. [환경 생성](#2-환경-생성)
3. [활성화 / 비활성화](#3-활성화--비활성화)
4. [기존 환경 복제](#4-기존-환경-복제)
5. [environment.yml로 내보내기 / 재현](#5-environmentyml로-내보내기--재현)
6. [ipykernel 연결 (Jupyter)](#6-ipykernel-연결-jupyter)
7. [패키지 관리 (conda vs pip)](#7-패키지-관리-conda-vs-pip)
8. [환경 목록 / 확인](#8-환경-목록--확인)
9. [삭제 및 정리](#9-삭제-및-정리)
10. [icard 실전 세팅 예시](#10-icard-실전-세팅-예시)
11. [트러블슈팅](#11-트러블슈팅)

---

## 1. 개념 한 줄 정리

- **환경(environment)**: 프로젝트별로 격리된 Python + 패키지 묶음. 프로젝트마다 따로 만들면 버전 충돌이 없다.
- **base**: conda 설치 시 기본으로 생기는 환경. 여기엔 가급적 패키지를 설치하지 않고 깨끗하게 둔다.
- **channel**: 패키지를 받아오는 저장소. `conda-forge`가 사실상 표준이며 패키지가 가장 많다.

> 팁: conda가 느리면 동일 명령을 그대로 쓰는 **mamba**(`conda install -n base mamba`)로 바꾸면 의존성 해석이 훨씬 빠르다. 아래 명령에서 `conda`를 `mamba`로만 바꿔 쓰면 된다.

---

## 2. 환경 생성

```bash
# 가장 기본: 이름 + 파이썬 버전만 지정
conda create -n icard python=3.11

# 만들면서 패키지까지 함께 설치 (충돌 해석이 더 안정적)
conda create -n icard python=3.11 numpy pip

# conda-forge 채널을 명시해서 생성
conda create -n icard -c conda-forge python=3.11
```

- `-n` (= `--name`): 환경 이름.
- 버전 고정은 `python=3.11`처럼 `=`로. 패키지도 `numpy=1.26`식으로 고정 가능.
- 특정 경로에 만들고 싶으면 이름 대신 `-p ./env`(프로젝트 폴더 안에 생성)를 쓴다.

---

## 3. 활성화 / 비활성화

```bash
conda activate icard      # 환경 진입 (프롬프트에 (icard) 표시됨)
conda deactivate          # 현재 환경 빠져나오기

# 터미널 열 때 자동으로 base가 켜지는 게 싫다면
conda config --set auto_activate_base false
```

---

## 4. 기존 환경 복제

같은 구성을 그대로 복사해 실험용으로 분리할 때 유용하다.

```bash
# icard 환경을 icard_test 라는 이름으로 복제
conda create -n icard_test --clone icard
```

- 같은 머신/OS 안에서의 복제에 적합하다.
- **다른 사람·다른 OS에 옮길 때**는 `--clone`보다 `environment.yml`(5번)이 재현성이 좋다. clone은 빌드 경로까지 복사해 OS가 다르면 깨질 수 있다.

---

## 5. environment.yml로 내보내기 / 재현

협업·배포·백업의 표준 방식. 환경을 파일 하나로 박제한다.

```bash
# 현재 환경 내보내기 (전체 — 정확하지만 OS 종속적)
conda activate icard
conda env export > environment.yml

# 권장: 직접 설치한 패키지만 기록 (OS 간 이식성이 좋음)
conda env export --from-history > environment.yml
```

```bash
# yml로부터 환경 새로 생성
conda env create -f environment.yml

# yml 변경 후 기존 환경에 반영 (없어진 패키지는 제거)
conda env update -f environment.yml --prune
```

`environment.yml` 예시 (icard용):

```yaml
name: icard
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pip
  - pip:
      - streamlit
      - torch
      - transformers
      - soundfile
      - sounddevice
      - pronouncing
```

> `--from-history`는 pip로 설치한 패키지를 잡지 못한다. pip 의존성이 있으면 위 예시처럼 `pip:` 블록을 직접 관리하거나, `pip freeze > requirements.txt`를 병행한다.

---

## 6. ipykernel 연결 (Jupyter)

만든 conda 환경을 Jupyter / VS Code 노트북에서 커널로 고를 수 있게 등록하는 단계.
두 가지 방법이 있다.

### 방법 A — 환경마다 수동 등록 (명시적)

```bash
conda activate icard
conda install ipykernel            # 또는 pip install ipykernel

python -m ipykernel install --user \
    --name icard \
    --display-name "Python (icard)"
```

- `--name`: 내부 식별자(중복 금지).
- `--display-name`: 노트북 커널 선택 목록에 보일 이름.

### 방법 B — 모든 환경 자동 연결 (추천)

base에 `nb_conda_kernels`를 한 번만 깔아두면, **ipykernel이 설치된 모든 conda 환경이 Jupyter에 자동으로 뜬다.** 환경을 새로 만들 때마다 수동 등록할 필요가 없다.

```bash
# base에서 1회만 설치
conda install -n base -c conda-forge nb_conda_kernels

# 각 환경에는 ipykernel만 깔려 있으면 자동 인식됨
conda install -n icard ipykernel
```

이후 `jupyter lab`을 base에서 실행하면 커널 목록에 `Python [conda env:icard]`처럼 자동 표시된다.

### 등록된 커널 확인 / 삭제

```bash
jupyter kernelspec list                 # 등록된 커널 목록
jupyter kernelspec uninstall icard      # 특정 커널 제거 (방법 A로 등록한 것)
```

---

## 7. 패키지 관리 (conda vs pip)

```bash
conda install -c conda-forge transformers   # conda로 설치
pip install streamlit                        # conda에 없거나 최신이 필요하면 pip
conda list                                   # 설치된 패키지 보기
conda update transformers                    # 특정 패키지 업데이트
conda update --all                           # 환경 전체 업데이트
```

**혼용 시 철칙**: conda로 설치할 것을 **먼저 다 깔고**, 그다음 pip을 쓴다. pip 먼저 깔고 conda를 다시 돌리면 환경이 깨지기 쉽다. 한 환경 안에서 같은 패키지를 conda와 pip로 중복 설치하지 않는다.

> torch는 플랫폼/GPU 조합을 타므로 보통 pip의 공식 설치 명령(혹은 conda-forge)을 그대로 쓰는 게 안전하다. CPU만 쓸 거면 `pip install torch`로 충분하다.

---

## 8. 환경 목록 / 확인

```bash
conda env list           # 또는 conda info --envs  → 전체 환경 목록 (* 가 현재 환경)
conda list -n icard      # 특정 환경의 패키지 목록
conda info               # conda 설정/버전/경로 정보
```

---

## 9. 삭제 및 정리

```bash
# 환경 통째로 삭제
conda env remove -n icard
# 또는
conda remove -n icard --all

# 캐시 / 안 쓰는 패키지 tarball 정리 (디스크 회복)
conda clean --all        # 캐시·미사용 패키지·index 모두 정리
conda clean --packages   # 미사용 패키지만
conda clean --tarballs   # 받은 압축본만
```

정기적으로 `conda clean --all`을 돌리면 수 GB가 회복되기도 한다.

---

## 10. icard 실전 세팅 예시

발음 평가 앱을 위한 환경을 처음부터 끝까지 세팅하는 전체 흐름.

```bash
# 1) 환경 생성
conda create -n icard -c conda-forge python=3.11 pip -y
conda activate icard

# 2) 핵심 패키지 설치 (conda 먼저, pip 나중)
conda install -c conda-forge numpy soundfile -y
pip install streamlit torch transformers sounddevice pronouncing

# 3) Jupyter 커널 자동 연결 (방법 B 기준)
conda install -n base -c conda-forge nb_conda_kernels -y
conda install ipykernel -y

# 4) 환경 박제 (백업/재현용)
conda env export --from-history > environment.yml

# 5) 앱 실행
streamlit run pronunciation_app.py
```

다른 PC에서 똑같이 재현할 때:

```bash
conda env create -f environment.yml
conda activate icard
pip install -r requirements.txt   # pip 패키지를 따로 관리했다면
```

---

## 11. 트러블슈팅

| 증상 | 해결 |
|------|------|
| `conda activate`가 안 먹힘 | 터미널에서 `conda init` 한 번 실행 후 셸 재시작 |
| 의존성 해석이 너무 느림 | `mamba` 사용 (`conda install -n base mamba` 후 `mamba`로 명령) |
| Jupyter에 환경이 안 보임 | 해당 환경에 `ipykernel` 설치 확인 + base에 `nb_conda_kernels` 설치 확인 |
| 환경이 꼬임 / pip·conda 충돌 | 환경 삭제 후 `environment.yml`로 새로 생성하는 게 가장 빠름 |
| 디스크 용량 부족 | `conda clean --all` 실행 |
| 커널이 목록에 남아있음 | `jupyter kernelspec uninstall <이름>` |

---

### 가장 자주 쓰는 명령 요약

```bash
conda create -n NAME python=3.11      # 생성
conda activate NAME                   # 진입
conda create -n NEW --clone OLD       # 복제
conda env export --from-history > environment.yml   # 내보내기
conda env create -f environment.yml   # 재현
python -m ipykernel install --user --name NAME --display-name "Python (NAME)"  # 커널 등록
conda env remove -n NAME              # 삭제
conda clean --all                     # 정리
```
