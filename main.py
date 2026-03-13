#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit Parser - CLI интерфейс для парсинга комментариев Reddit
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
    print(f"    {c('RED','2)')}  Выход")
    print()
    return(int(input("Режим: ")))


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

if __name__ == "__main__":
    # Добавляем текущую директорию в путь
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print_header()
    mode = print_menu()
    if mode == 1:
        run_cli_interface()