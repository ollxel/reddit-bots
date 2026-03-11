#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit Parser - CLI и Web интерфейс для парсинга комментариев Reddit
"""
import os
import sys

# Fix Windows encoding - simplified approach
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')
    except Exception:
        pass

import subprocess
import webbrowser
import time
from pathlib import Path


def c(color, text):
    """Apply color to text"""
    colors = {
        'BLUE': '\033[94m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'BOLD': '\033[1m',
        'END': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['END']}"


def print_header():
    """Выводит заголовок приложения"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(c('CYAN', c('BOLD', """
============================================================
                    Reddit Parser v2.0
              Парсинг комментариев Reddit
============================================================
    """)))


def print_menu():
    """Выводит главное меню"""
    print(c('BLUE', 'Выберите режим работы:'))
    print()
    print(f"    {c('GREEN','1)')}  CLI интерфейс")
    print("           - Работа в терминале")
    print("           - Ввод API ключа и параметров вручную")
    print()
    print(f"    {c('PURPLE','2)')}  Web интерфейс")
    print("           - Красивый веб-интерфейс")
    print("           - Анимация и визуализация результатов")
    print("           - Tailwind CSS")
    print()
    print(f"    {c('RED','3)')}  Выход")
    print()


def run_cli_interface():
    """Запускает CLI интерфейс для парсинга"""
    print(c('BLUE', '\n' + '='*60))
    print("Запуск CLI интерфейса...")
    print('='*60 + '\n')

    # Импортируем модули парсера
    from main_parser import RedditParser, SentimentAnalyzer
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Шаг 1: Ввод API ключа
    print(c('YELLOW', '\nШаг 1: Настройка OpenRouter API'))
    print("-" * 40)
    
    # Проверяем есть ли ключ в .env
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        print(f"Найден ключ в .env файле: {c('GREEN','OK')}")
        use_existing = input("Использовать существующий ключ? (y/n): ").strip().lower()
        if use_existing in ['y', 'yes', 'д', 'да']:
            api_key = env_key
            model = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
        else:
            api_key = input("Введите ваш OpenRouter API ключ: ").strip()
            model = input("Введите модель (по умолчанию: arcee-ai/trinity-large-preview:free): ").strip()
            if not model:
                model = "arcee-ai/trinity-large-preview:free"
    else:
        api_key = input("Введите ваш OpenRouter API ключ: ").strip()
        model = input("Введите модель (по умолчанию: arcee-ai/trinity-large-preview:free): ").strip()
        if not model:
            model = "arcee-ai/trinity-large-preview:free"

    # Вопрос об использовании sentiment анализа
    use_sentiment = api_key and input("\nВключить анализ тональности комментариев? (y/n): ").strip().lower() in ['y', 'yes', 'д', 'да']

    # Шаг 2: Выбор режима парсинга
    print(c('YELLOW', '\nШаг 2: Выбор режима парсинга'))
    print("-" * 40)
    print("1) Парсить последние посты сабреддита")
    print("2) Парсить конкретный пост по ссылке")
    
    parse_mode = input("Выберите режим (1/2): ").strip()

    # Шаг 3: Ввод параметров
    print(c('YELLOW', '\nШаг 3: Параметры парсинга'))
    print("-" * 40)
    
    target_comments = int(input("Количество комментариев для сбора (по умолчанию 100): ").strip() or "100")
    
    # Создаём парсер
    parser = RedditParser(
        user_agent="RedditDataCollector/1.0 (Educational)",
        run_sentiment=use_sentiment
    )
    
    # Если есть API ключ, создаём SentimentAnalyzer
    if use_sentiment and api_key:
        parser.sentiment = SentimentAnalyzer(api_key=api_key, model=model)
        print(f"{c('GREEN','OK')} SentimentAnalyzer готов (модель: {model})")

    # Запуск парсинга
    print(c('GREEN', '\n' + '='*60))
    print("Начинаем парсинг...")
    print('='*60 + '\n')

    if parse_mode == "1":
        # Парсинг сабреддита
        subreddit = input("Введите название сабреддита (без r/): ").strip()
        category = input("Категория (hot/new/top/rising, по умолчанию hot): ").strip() or "hot"
        
        if category == "top":
            time_filter = input("Временной фильтр (hour/day/week/month/year/all, по умолчанию week): ").strip() or "week"
        else:
            time_filter = "week"
        
        df = parser.parse_subreddit_comments(
            subreddit,
            posts_limit=10,
            category=category,
            time_filter=time_filter,
            target_comments=target_comments
        )
    else:
        # Парсинг по ссылке
        post_url = input("Введите ссылку на пост (https://reddit.com/r/...): ").strip()
        
        df = parser.parse_post_comments(
            post_url,
            target_comments=target_comments
        )

    # Вывод результатов
    if not df.empty:
        print(c('GREEN', '\n' + '='*60))
        print(f"Парсинг завершён! Собрано {len(df)} уникальных пользователей")
        print('='*60)
        
        stats = parser.get_stats()
        print("\n--- Статистика ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(c('RED', '\nДанные не собраны'))

    input("\nНажмите Enter для продолжения...")


def run_web_interface():
    """Запускает Web интерфейс"""
    print(c('PURPLE', '\n' + '='*60))
    print("Запуск Web интерфейса...")
    print('='*60 + '\n')

    # Проверяем установлены ли зависимости Node.js
    web_server_dir = Path(__file__).parent / "web-server"
    
    # Установка зависимостей Node.js если нужно
    print("Проверка зависимостей Node.js...")
    if not (web_server_dir / "node_modules").exists():
        print("Установка зависимостей Node.js...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=str(web_server_dir),
                check=True,
                capture_output=True
            )
            print(c('GREEN', 'OK') + " Зависимости установлены")
        except subprocess.CalledProcessError as e:
            print(c('RED', 'Ошибка установки зависимостей:'))
            print(e.stderr.decode() if e.stderr else str(e))
            input("Нажмите Enter для выхода...")
            return
        except FileNotFoundError:
            print(c('RED', 'Ошибка: npm не найден.'))
            print("Пожалуйста, установите Node.js с сайта https://nodejs.org/")
            input("Нажмите Enter для выхода...")
            return

    # Запуск Node.js сервера
    print("\nЗапуск веб-сервера...")
    print(c('YELLOW', 'Сервер запускается на http://localhost:3000'))
    
    try:
        # Запускаем сервер
        server_process = subprocess.Popen(
            ["node", "server.js"],
            cwd=str(web_server_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Ждём запуска сервера
        time.sleep(2)
        
        # Открываем браузер
        print(c('GREEN', '\nОткрытие браузера...'))
        webbrowser.open("http://localhost:3000")
        
        print(c('CYAN', '\n' + '='*60))
        print("Web интерфейс запущен!")
        print(f"Откройте в браузере: http://localhost:3000")
        print('='*60)
        print(c('YELLOW', '\nНажмите Ctrl+C для остановки сервера'))
        
        # Ждём пока сервер работает
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print(c('RED', '\nОстановка сервера...'))
            server_process.terminate()
            server_process.wait()
            
    except FileNotFoundError:
        print(c('RED', 'Ошибка: node не найден.'))
        print("Пожалуйста, установите Node.js с сайта https://nodejs.org/")
        input("Нажмите Enter для выхода...")
    except Exception as e:
        print(c('RED', f'Ошибка запуска сервера: {e}'))
        input("Нажмите Enter для выхода...")


def main():
    """Главная функция"""
    while True:
        print_header()
        print_menu()
        
        choice = input(c('BOLD', '\nВаш выбор: ')).strip()
        
        if choice == "1":
            run_cli_interface()
        elif choice == "2":
            run_web_interface()
            # После запуска веб-интерфейса возвращаемся в меню
            input("\nНажмите Enter для возврата в меню...")
        elif choice == "3" or choice.lower() in ['exit', 'quit', 'выход', 'q']:
            print(c('CYAN', '\nДо свидания!\n'))
            break
        else:
            print(c('RED', '\nНеверный выбор. Попробуйте снова.'))
            time.sleep(1)


if __name__ == "__main__":
    # Добавляем текущую директорию в путь
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
