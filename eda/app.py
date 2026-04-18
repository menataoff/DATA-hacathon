import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# ------------------ НАСТРОЙКА СТРАНИЦЫ ------------------
st.set_page_config(page_title="Прогноз оттока + Рекомендации v8", layout="wide", page_icon="📊")
st.title("📊 Прогнозирование оттока клиентов и рекомендации v8")
st.markdown("### Интерактивная платформа для оценки риска ухода и персонализированных предложений")

# ------------------ ЗАГРУЗКА МОДЕЛИ И ДАННЫХ (ОТТОК) ------------------
@st.cache_resource
def load_model():
    if os.path.exists('best_churn_model.pkl'):
        return joblib.load('best_churn_model.pkl')
    elif os.path.exists('../best_churn_model.pkl'):
        return joblib.load('../best_churn_model.pkl')
    else:
        st.error("❌ Модель оттока не найдена. Загрузите best_churn_model.pkl")
        st.stop()

@st.cache_data
def load_features():
    if os.path.exists('features_full.csv'):
        return pd.read_csv('features_full.csv')
    elif os.path.exists('../features_full.csv'):
        return pd.read_csv('../features_full.csv')
    else:
        st.error("❌ Файл features_full.csv не найден")
        st.stop()

model = load_model()
df = load_features()
feature_cols = model.feature_name_
mean_vals = df[feature_cols].mean().to_dict()

imp_df = None
if os.path.exists('feature_importance_p3.csv'):
    imp_df = pd.read_csv('feature_importance_p3.csv')
elif os.path.exists('../feature_importance_p3.csv'):
    imp_df = pd.read_csv('../feature_importance_p3.csv')

# ------------------ ЗАГРУЗКА ДАННЫХ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ v8 ------------------
@st.cache_data
def load_v8_data():
    data = {}
    files = ['v8_recommendations.csv', 'metrics_comparison_v8.csv',
             'segment_comparison_v8.csv', 'metric_selection_v8.csv', 'v8_summary.json']
    for f in files:
        if os.path.exists(f):
            if f.endswith('.json'):
                with open(f, 'r', encoding='utf-8') as jf:
                    data[f.replace('.json', '')] = json.load(jf)
            else:
                data[f.replace('.csv', '')] = pd.read_csv(f)
        elif os.path.exists(f'../{f}'):
            if f.endswith('.json'):
                with open(f'../{f}', 'r', encoding='utf-8') as jf:
                    data[f.replace('.json', '')] = json.load(jf)
            else:
                data[f.replace('.csv', '')] = pd.read_csv(f'../{f}')
    return data

v8_data = load_v8_data()

# ------------------ БОКОВАЯ ПАНЕЛЬ ------------------
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел",
    ["Главная", "Исследование данных", "Кластерный анализ", "Модель оттока",
     "Интерактивное предсказание", "Гиперпараметры (демо)",
     "Рекомендации v8", "Ручной профиль (v8)"]
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
    st.info("💡 *Все выводы основаны на анализе реальных данных (синтетических, но с сохранением всех закономерностей).*")

# ------------------ ИССЛЕДОВАНИЕ ДАННЫХ ------------------
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

# ------------------ КЛАСТЕРНЫЙ АНАЛИЗ ------------------
elif page == "Кластерный анализ":
    st.header("👥 Кластерный анализ клиентской базы")
    st.markdown("""
    Мы разделили всех клиентов на **4 поведенческих кластера** с помощью метода K‑means.  
    Оптимальное число кластеров выбрано по методу локтя и силуэта (см. график).  
    Каждый кластер требует своей стратегии взаимодействия.
    """)
    # Отображение графиков, если они есть
    if os.path.exists('cluster_analysis.png'):
        st.image('cluster_analysis.png', caption="Метод локтя (инерция) и силуэт (качество кластеризации)", use_container_width=True)
    elif os.path.exists('../cluster_analysis.png'):
        st.image('../cluster_analysis.png', caption="Метод локтя и силуэт", use_container_width=True)
    else:
        st.warning("Файл cluster_analysis.png не найден. Вы можете сгенерировать его из ноутбука task2.")

    st.subheader("📋 Характеристики кластеров")
    st.markdown("""
    | Кластер | Размер | recency | frequency | monetary | return_rate | days_since_last_event | Интерпретация | Стратегия |
    |---------|--------|---------|-----------|----------|-------------|----------------------|---------------|------------|
    | **0** | 8 223 | 980 | 0.10 | 15 | **1.00** | 559 | **Проблемные возвраты** – все покупки возвращены, давно не активны | Не тратить ресурсы на удержание |
    | **1** | 43 472 | 1046 | 0.12 | 21 | 0.00 | 704 | **Спящие / неактивные** – почти не покупали, давно ушли | Кампания реактивации (скидки, напоминания) |
    | **2** | 14 278 | 817 | 0.64 | 218 | 0.20 | 422 | **Активные с возвратами** – много покупают, но часто возвращают | Улучшить качество, предлагать товары с низкими возвратами |
    | **3** | 14 048 | 350 | **1.13** | **265** | 0.02 | **316** | **Активные и прибыльные** – самые ценные клиенты | Программы лояльности, бонусы, удержание |
    """)
    st.success("**Рекомендации:** Для кластера 1 (спящие) – запустить email‑кампанию с персональными скидками. Для кластера 2 – пересмотреть ассортимент и качество товаров.")

# ------------------ МОДЕЛЬ ОТТОКА ------------------
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
        st.caption("ROC-кривая показывает, как часто модель правильно идентифицирует отточных клиентов. Площадь под кривой (AUC) = 0.772 – хороший результат.")
    elif os.path.exists('../roc_curve.png'):
        st.image('../roc_curve.png', use_container_width=True)
    else:
        st.info("Файл roc_curve.png не найден. Сгенерируйте его из ноутбука.")

    if os.path.exists('pr_curve.png'):
        st.subheader("📈 PR-кривая")
        st.image('pr_curve.png', use_container_width=True)
        st.caption("PR-кривая более чувствительна к дисбалансу. Высокое значение PR-AUC = 0.992 подтверждает, что модель отлично ранжирует клиентов по риску.")

# ------------------ ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ ------------------
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
    st.markdown("💡 *Попробуйте установить `days_since_last_event = 600` и `recency = 800` – вероятность оттока станет очень высокой.*")

# ------------------ ГИПЕРПАРАМЕТРЫ (ДЕМО) ------------------
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
    penalty = (abs(n_estimators - 295) / 500) * 0.05 + (abs(max_depth - 7) / 15) * 0.03 + (abs(learning_rate - 0.072) / 0.2) * 0.04
    sim_auc = max(0.65, min(0.78, base_auc - penalty))

    st.metric("Симулированный ROC-AUC", f"{sim_auc:.4f}")
    st.progress((sim_auc - 0.65) / 0.15)
    st.caption("**Лучшие параметры по результатам Optuna:** n_estimators=295, max_depth=7, learning_rate=0.072, subsample=0.918, colsample_bytree=0.833.")
    st.info("При реальном обучении модель с такими параметрами дала ROC-AUC = 0.772. Дальнейшее увеличение сложности (больше деревьев, глубже) приводит к переобучению и снижению метрики на тесте.")

# ------------------ РЕКОМЕНДАЦИИ v8 (ПРОСМОТР ТАБЛИЦЫ, СРАВНЕНИЕ МЕТОДОВ) ------------------
elif page == "Рекомендации v8":
    st.header("🎁 Рекомендательная система v8")
    st.markdown("""
    **Гибридная модель** – объяснимый алгоритм, который опирается на:
    - силу товара (`item_score`),
    - интерес пользователя к категориям и переходы между ними,
    - риск ухода (`risk_score`),
    - ценовое соответствие (`price_fit`),
    - безопасный резерв (cold start / safe explore).

    **Главные метрики выбора модели:**
    - `CategoryHit@10` – доля пользователей, у которых следующая реальная категория покупки попала в top-10 рекомендаций.
    - `RetentionCategoryHit@10` – то же, но с весом `1 + risk_score`, т.е. важнее попасть в интерес для рискованных клиентов.
    """)

    if 'v8_recommendations' not in v8_data:
        st.warning("Файлы рекомендаций v8 не найдены. Пожалуйста, поместите в папку `v8_recommendations.csv`, `metrics_comparison_v8.csv` и др.")
    else:
        rec_df = v8_data['v8_recommendations']
        user_ids = sorted(rec_df['user_id'].unique())
        selected_user = st.selectbox("Выберите пользователя", user_ids)
        user_recs = rec_df[rec_df['user_id'] == selected_user].sort_values('rank')
        st.subheader(f"📌 Рекомендации для {selected_user}")
        st.dataframe(user_recs[['rank', 'product_id', 'recommended_category', 'final_score', 'source_block']], use_container_width=True)

        if 'metrics_comparison_v8' in v8_data:
            st.subheader("📊 Сравнение методов")
            metrics_df = v8_data['metrics_comparison_v8']
            st.dataframe(metrics_df, use_container_width=True)

            fig = go.Figure()
            for metric in ['CategoryHit@10', 'RetentionCategoryHit@10', 'AvgQuality@10', 'AvgABC@10']:
                if metric in metrics_df.columns:
                    fig.add_trace(go.Bar(name=metric, x=metrics_df['model'], y=metrics_df[metric]))
            fig.update_layout(title="Сравнение метрик", barmode='group', height=500)
            st.plotly_chart(fig, use_container_width=True)

        if 'segment_comparison_v8' in v8_data:
            st.subheader("📈 Качество по риск-сегментам")
            seg = v8_data['segment_comparison_v8']
            fig = px.bar(seg, x='risk_segment', y='CategoryHit_10', color='model',
                         title="CategoryHit@10 по сегментам риска", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        if 'metric_selection_v8' in v8_data:
            st.subheader("🔍 Выбор главных метрик")
            st.dataframe(v8_data['metric_selection_v8'], use_container_width=True)

        # Блок с формулами (кратко)
        with st.expander("📐 Основные формулы v8 (справочно)"):
            st.markdown(r"""
            **Сила товара:**  
            $\text{item_score} = 0.45 \cdot \text{pop\_norm} + 0.20 \cdot \text{revenue\_norm} + 0.20 \cdot \text{abc\_score} + 0.15 \cdot \text{quality\_score}$

            **Интерес пользователя к категории:**  
            $\text{cat\_pref\_score}_{u,c} = \frac{0.45 \cdot \text{cnt} + 2.50 \cdot \text{spend\_share} + 0.35 \cdot \text{recency\_sum}}{\max_c(...)}$

            **Переходы:**  
            $\text{transition\_prob}_{a\to b} = \frac{N_{a\to b}}{\sum_{b'} N_{a\to b'}}$, берутся top-5.

            **Итоговая сила категории:**  
            $\text{cat\_score} = 0.45 \cdot \text{cat\_pref\_score} + (0.35 + 0.15(1-\text{risk\_score})) \cdot \text{transition\_score}$

            **Балл товара в историко-переходном блоке:**  
            $\text{score}^{(hist)} = 0.42 \cdot \text{cat\_score} + (0.22+0.08\cdot\text{risk})\cdot\text{pop\_norm} + 0.12\cdot\text{revenue\_norm} + (0.12+0.08\cdot\text{risk})\cdot\text{quality} + 0.08\cdot\text{price\_fit} + 0.06\cdot\text{abc}$

            **Безопасный блок (explore):**  
            $\text{score}^{(explore)} = 0.30\cdot\text{pop\_norm} + 0.20\cdot\text{revenue\_norm} + 0.20\cdot\text{quality} + 0.12\cdot\text{price\_fit} + 0.08\cdot\text{abc} + 0.10\cdot\text{risk}$
            """)

# ------------------ РУЧНОЙ ПРОФИЛЬ (v8) ------------------
elif page == "Ручной профиль (v8)":
    st.header("🎛️ Ручной режим: задайте профиль клиента")
    st.markdown("""
    Укажите до трёх категорий интереса, вероятность ухода и уже купленные товары.  
    Система подберёт рекомендации на основе логики **v8_transition_retention**.
    """)

    if 'v8_recommendations' not in v8_data:
        st.warning("Файл v8_recommendations.csv не найден. Ручной режим недоступен.")
    else:
        rec_full = v8_data['v8_recommendations']
        all_categories = sorted(rec_full['recommended_category'].dropna().unique())

        col1, col2 = st.columns(2)
        with col1:
            cat1 = st.selectbox("Категория 1 (приоритет)", options=["—"] + all_categories, index=0)
            cat2 = st.selectbox("Категория 2", options=["—"] + all_categories, index=0)
            cat3 = st.selectbox("Категория 3", options=["—"] + all_categories, index=0)
        with col2:
            churn_prob = st.slider("Вероятность ухода (churn_probability)", 0.0, 1.0, 0.5, 0.01)
            purchased_raw = st.text_input("Уже купленные product_id (через запятую)", placeholder="123, 456, 789")
            purchased_ids = [int(x.strip()) for x in purchased_raw.split(",") if x.strip().isdigit()]

        if st.button("Получить рекомендации", type="primary"):
            # Определяем риск-сегмент
            if churn_prob <= 0.67:
                risk_seg = "low_risk"
            elif churn_prob <= 0.90:
                risk_seg = "medium_risk"
            else:
                risk_seg = "high_risk"

            # Собираем выбранные категории
            cats = [c for c in [cat1, cat2, cat3] if c != "—"]
            # Создаём упрощённый риск-сегмент в DataFrame
            df_temp = rec_full.copy()
            df_temp['risk_segment_sim'] = pd.cut(df_temp['churn_probability'], bins=[-1, 0.67, 0.90, 2],
                                                 labels=['low_risk', 'medium_risk', 'high_risk'])
            if cats:
                mask = df_temp['recommended_category'].isin(cats) & (df_temp['risk_segment_sim'] == risk_seg)
                candidates = df_temp[mask].sort_values(['rank', 'final_score'], ascending=[True, False])
            else:
                candidates = df_temp.sort_values(['rank', 'final_score'], ascending=[True, False])
            # Исключаем уже купленные
            candidates = candidates[~candidates['product_id'].isin(purchased_ids)]
            top = candidates.drop_duplicates('product_id').head(10).copy()
            top['rank'] = range(1, len(top)+1)

            st.subheader("📌 Рекомендации")
            st.dataframe(top[['rank', 'product_id', 'recommended_category', 'final_score', 'quality_score', 'abc_score', 'source_block']], use_container_width=True)

            # Визуализация баллов
            fig = px.bar(top, x='rank', y='final_score', hover_data=['product_id', 'recommended_category'],
                         title=f"Итоговый балл рекомендаций (риск-сегмент: {risk_seg})")
            st.plotly_chart(fig, use_container_width=True)

# ------------------ ПОДВАЛ ------------------
st.caption("© Комплексное решение – прогноз оттока + рекомендации v8. Модель оттока обучена без утечки данных.")