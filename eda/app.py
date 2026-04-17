import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import lightgbm as lgb

# ------------------ НАСТРОЙКА СТРАНИЦЫ ------------------
st.set_page_config(page_title="Прогноз оттока клиентов", layout="wide", page_icon="📊")
st.title("📊 Прогнозирование оттока клиентов")
st.markdown("### Интерактивная платформа для оценки риска ухода клиентов и исследования ключевых факторов")


# ------------------ ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ------------------
@st.cache_resource
def load_model():
    if os.path.exists('best_churn_model.pkl'):
        return joblib.load('best_churn_model.pkl')
    elif os.path.exists('../best_churn_model.pkl'):
        return joblib.load('../best_churn_model.pkl')
    else:
        st.error("Модель не найдена. Загрузите best_churn_model.pkl")
        st.stop()


@st.cache_data
def load_data():
    if os.path.exists('features_full.csv'):
        return pd.read_csv('features_full.csv')
    elif os.path.exists('../features_full.csv'):
        return pd.read_csv('../features_full.csv')
    else:
        st.error("Файл features_full.csv не найден")
        st.stop()


model = load_model()
df = load_data()
feature_cols = model.feature_name_
mean_vals = df[feature_cols].mean().to_dict()

if os.path.exists('feature_importance_p3.csv'):
    imp_df = pd.read_csv('feature_importance_p3.csv')
else:
    imp_df = None

# ------------------ БОКОВАЯ ПАНЕЛЬ НАВИГАЦИИ ------------------
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел",
    ["Главная", "Исследование данных", "Кластерный анализ", "Модель оттока", "Интерактивное предсказание",
     "Гиперпараметры (демо)"]
)

# ------------------ СТРАНИЦА "ГЛАВНАЯ" ------------------
if page == "Главная":
    st.header("🎯 Цель проекта")
    st.markdown("""
    **Борьба с оттоком клиентов** – одна из ключевых задач любого бизнеса.  
    Мы разработали комплексное аналитическое решение, которое позволяет:
    - **Диагностировать проблемы** клиентов с помощью разведочного анализа и сегментации.
    - **Прогнозировать риск оттока** с высокой точностью (ROC-AUC = 0.772, PR-AUC = 0.992).
    - **Рекомендовать персонализированные предложения** для удержания клиентов с высоким риском.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Клиентов в базе", f"{len(df):,}")
    col2.metric("Доля оттока (обучение)", "97.7%", help="В обучающей выборке (сильный дисбаланс)")
    col3.metric("ROC-AUC модели", "0.772", help="LightGBM с class_weight='balanced'")

    st.subheader("📈 Ключевые инсайты для бизнеса")
    st.markdown("""
    - **Главный триггер оттока** – **давность последнего визита на сайт** (`days_since_last_event`).  
      Если клиент не заходил более 90 дней, вероятность оттока резко возрастает.
    - **География** (`city`, `state`) – второй по важности фактор. В некоторых регионах отток выше из‑за проблем с логистикой или сервисом.
    - **Активность на сайте** (`total_events`) важнее, чем сумма покупок. Клиент, который часто просматривает товары, реже уходит.
    - **Возраст** также влияет: в наших данных клиенты 40–50 лет более лояльны, чем молодые.
    - **Высокая доля возвратов** – сигнал недовольства. Клиенты с возвратами более 20% имеют повышенный риск.
    """)
    st.info(
        "💡 *Все выводы основаны на анализе реальных данных (синтетических, но с сохранением всех закономерностей).*")

# ------------------ СТРАНИЦА "ИССЛЕДОВАНИЕ ДАННЫХ" ------------------
elif page == "Исследование данных":
    st.header("🔍 Разведочный анализ данных")
    st.markdown("""
    Здесь вы можете изучить распределения ключевых признаков и их корреляции.  
    **Что важно:**  
    - `recency` (дней с последней покупки) у большинства клиентов велико – они давно не покупали.
    - `days_since_last_event` (дней с последнего действия на сайте) также имеет высокие значения.
    - Корреляционная матрица показывает, что `recency` и `days_since_last_event` сильно связаны (логично).
    """)
    selected_feat = st.selectbox("Выберите признак для просмотра распределения", feature_cols[:20])
    fig = px.histogram(df, x=selected_feat, title=f"Распределение {selected_feat}", nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Корреляционная матрица (топ-10 важных признаков)")
    if imp_df is not None:
        top10_feats = imp_df.head(10)['feature'].tolist()
        corr = df[top10_feats].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Корреляции между важнейшими признаками")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Файл с важностью признаков не загружен.")

# ------------------ СТРАНИЦА "КЛАСТЕРНЫЙ АНАЛИЗ" ------------------
elif page == "Кластерный анализ":
    st.header("👥 Кластерный анализ клиентской базы")
    st.markdown("""
    Мы разделили всех клиентов на **4 поведенческих кластера** с помощью метода K‑means.  
    Оптимальное число кластеров выбрано по методу локтя и силуэта (см. график).  
    Каждый кластер требует своей стратегии взаимодействия.
    """)
    if os.path.exists('cluster_analysis.png'):
        st.image('cluster_analysis.png', caption="Метод локтя (инерция) и силуэт (качество кластеризации)",
                 use_container_width=True)
    elif os.path.exists('../cluster_analysis.png'):
        st.image('../cluster_analysis.png', caption="Метод локтя и силуэт", use_container_width=True)
    else:
        st.warning("Файл cluster_analysis.png не найден.")

    st.subheader("📋 Характеристики кластеров")
    st.markdown("""
    | Кластер | Размер | recency | frequency | monetary | return_rate | days_since_last_event | Интерпретация | Стратегия |
    |---------|--------|---------|-----------|----------|-------------|----------------------|---------------|------------|
    | **0** | 8 223 | 980 | 0.10 | 15 | **1.00** | 559 | **Проблемные возвраты** – все покупки возвращены, давно не активны | Не тратить ресурсы на удержание |
    | **1** | 43 472 | 1046 | 0.12 | 21 | 0.00 | 704 | **Спящие / неактивные** – почти не покупали, давно ушли | Кампания реактивации (скидки, напоминания) |
    | **2** | 14 278 | 817 | 0.64 | 218 | 0.20 | 422 | **Активные с возвратами** – много покупают, но часто возвращают | Улучшить качество, предлагать товары с низкими возвратами |
    | **3** | 14 048 | 350 | **1.13** | **265** | 0.02 | **316** | **Активные и прибыльные** – самые ценные клиенты | Программы лояльности, бонусы, удержание |
    """)
    st.success(
        "**Рекомендации:** Для кластера 1 (спящие) – запустить email‑кампанию с персональными скидками. Для кластера 2 – пересмотреть ассортимент и качество товаров.")

# ------------------ СТРАНИЦА "МОДЕЛЬ ОТТОКА" ------------------
elif page == "Модель оттока":
    st.header("🧠 Модель прогнозирования оттока")
    st.markdown("""
    **Использованный алгоритм:** LightGBM с `class_weight='balanced'` (компенсация сильного дисбаланса классов).  
    **Почему LightGBM?** – быстрый, хорошо работает с большими данными, даёт интерпретируемые важности признаков.  
    **Предобработка:** временное разделение (обучение на данных до 2025-12-11, тест – на более поздних), чтобы избежать утечки.
    """)
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", "0.772", help="Чем ближе к 1, тем лучше модель разделяет классы")
    col2.metric("PR-AUC", "0.992", help="Более честная метрика при сильном дисбалансе")

    st.subheader("📊 Важность признаков (топ-15)")
    if imp_df is not None:
        fig_imp = px.bar(imp_df.head(15), x='importance', y='feature', orientation='h',
                         title="Вклад каждого признака в предсказание оттока")
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("""
        **Что означает `state`?**  
        `state` – это штат или регион (в наших данных – закодированное число).  
        Высокая важность говорит о том, что география сильно влияет на отток. Возможные причины: проблемы с доставкой, разная экономическая ситуация, культурные особенности. Бизнесу стоит проанализировать регионы с наибольшим оттоком.
        """)
    else:
        st.info("Файл важности признаков не найден.")

    st.subheader("📉 ROC-кривая")
    if os.path.exists('roc_curve.png'):
        st.image('roc_curve.png', use_container_width=True)
        st.caption(
            "ROC-кривая показывает, как часто модель правильно идентифицирует отточных клиентов. Площадь под кривой (AUC) = 0.772 – хороший результат.")
    elif os.path.exists('../roc_curve.png'):
        st.image('../roc_curve.png', use_container_width=True)
    else:
        st.info("Файл roc_curve.png не найден. Сгенерируйте его из ноутбука.")

    if os.path.exists('pr_curve.png'):
        st.subheader("📈 PR-кривая")
        st.image('pr_curve.png', use_container_width=True)
        st.caption(
            "PR-кривая более чувствительна к дисбалансу. Высокое значение PR-AUC = 0.992 подтверждает, что модель отлично ранжирует клиентов по риску.")

# ------------------ СТРАНИЦА "ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ" ------------------
elif page == "Интерактивное предсказание":
    st.header("🎚️ Предскажите вероятность оттока")
    st.markdown("""
    **Как пользоваться:**  
    Двигайте ползунки, чтобы изменить параметры гипотетического клиента.  
    Модель мгновенно рассчитает вероятность оттока.  
    *Чем выше `days_since_last_event` и `recency`, тем выше риск.*  
    """)
    key_features = ['days_since_last_event', 'recency', 'monetary', 'total_events', 'age', 'return_rate']
    user_input = {}
    cols = st.columns(2)
    for i, feat in enumerate(key_features):
        if feat in feature_cols:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            with cols[i % 2]:
                user_input[feat] = st.slider(
                    f"{feat}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Диапазон в данных: {min_val:.1f} – {max_val:.1f}"
                )
    input_dict = mean_vals.copy()
    for feat in key_features:
        if feat in input_dict:
            input_dict[feat] = user_input[feat]
    input_df = pd.DataFrame([input_dict])[feature_cols]

    proba = model.predict_proba(input_df)[0, 1]
    risk = "Высокий" if proba > 0.7 else "Средний" if proba > 0.3 else "Низкий"
    color = "red" if proba > 0.7 else "orange" if proba > 0.3 else "green"

    st.markdown("## Результат")
    col1, col2, col3 = st.columns(3)
    col1.metric("Вероятность оттока", f"{proba:.2%}")
    col2.markdown(f"**Уровень риска:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
    col3.markdown(
        f"**Рекомендация:** {'🚨 Срочное удержание (персональная скидка, колл)' if proba > 0.7 else '⚡ Мониторинг, можно отправить email' if proba > 0.3 else '✅ Стабильный клиент, поддерживать лояльность'}")

    st.markdown("---")
    st.markdown(
        "💡 *Попробуйте установить `days_since_last_event = 600` и `recency = 800` – вероятность оттока станет очень высокой.*")

# ------------------ СТРАНИЦА "ГИПЕРПАРАМЕТРЫ (ДЕМО)" ------------------
elif page == "Гиперпараметры (демо)":
    st.header("⚙️ Влияние гиперпараметров на качество модели")
    st.markdown("""
    Этот раздел **симулирует** изменение ROC-AUC при изменении гиперпараметров (без реального переобучения, на основе эмпирических зависимостей).  
    В реальном проекте мы использовали **Optuna** для поиска лучших параметров.  
    Вы можете поэкспериментировать и увидеть, как отклонение от оптимума снижает качество.
    """)
    n_estimators = st.slider("Количество деревьев (n_estimators)", 50, 500, 295, step=10)
    max_depth = st.slider("Максимальная глубина (max_depth)", 3, 15, 7, step=1)
    learning_rate = st.slider("Скорость обучения (learning_rate)", 0.01, 0.2, 0.072, step=0.005)

    base_auc = 0.772
    penalty = (abs(n_estimators - 295) / 500) * 0.05 + (abs(max_depth - 7) / 15) * 0.03 + (
                abs(learning_rate - 0.072) / 0.2) * 0.04
    sim_auc = max(0.65, min(0.78, base_auc - penalty))

    st.metric("Симулированный ROC-AUC", f"{sim_auc:.4f}")
    st.progress((sim_auc - 0.65) / 0.15)
    st.caption(
        "**Лучшие параметры по результатам Optuna:** n_estimators=295, max_depth=7, learning_rate=0.072, subsample=0.918, colsample_bytree=0.833.")
    st.info(
        "При реальном обучении модель с такими параметрами дала ROC-AUC = 0.772. Дальнейшее увеличение сложности (больше деревьев, глубже) приводит к переобучению и снижению метрики на тесте.")