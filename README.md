# Прогноз оттока клиентов — интерактивный сайт

## Локальный запуск (рекомендуемый способ)

### Требования
- Python 3.12 или выше
- Git

### Шаги для запуска

1. **Клонируйте репозиторий**
   ```bash
   git clone https://github.com/ваш_аккаунт/ваш_репозиторий.git
   cd ваш_репозиторий
   ```
2. **Создайте виртуальное окружение (рекомендуется)**
   ```bash
    python -m venv venv
    source venv/bin/activate      # для Linux/Mac
    venv\Scripts\activate         # для Windows
   ```
3. **Установите зависимости**
    ```bash
   pip install -r requirements.txt
   ```
4. **Запустите приложение**
    ```bash
    cd eda
    streamlit run eda/app.py
    ```
