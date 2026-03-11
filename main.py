#!/usr/bin/env python3
"""
Reddit Parser - CLI и Web интерфейс для парсинга комментариев Reddit
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Цвета для CLI
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header():
    """Выводит заголовок приложения"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                    Reddit Parser v2.0                         ║
║              Парсинг комментариев Reddit                      ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.END}
    """)


def print_menu():
    """Выводит главное меню"""
    print(f"""
{Colors.BLUE}Выберите режим работы:{Colors.END}

    {Colors.GREEN}1){Colors.END}  🔵 CLI интерфейс
           - Работа в терминале
           - Ввод API ключа и параметров вручную

    {Colors.PURPLE}2){Colors.END}  🟣 Web интерфейс
           - Красивый веб-интерфейс
           - Анимация и визуализация результатов
           - Tailwind CSS

    {Colors.RED}3){Colors.END}  ❌ Выход
    """)


def run_cli_interface():
    """Запускает CLI интерфейс для парсинга"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("Запуск CLI интерфейса...")
    print(f"{'='*60}{Colors.END}\n")

    # Импортируем модули парсера
    from main_parser import RedditParser, SentimentAnalyzer
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Шаг 1: Ввод API ключа
    print(f"\n{Colors.YELLOW}Шаг 1: Настройка OpenRouter API{Colors.END}")
    print("-" * 40)
    
    # Проверяем есть ли ключ в .env
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        print(f"Найден ключ в .env файле: {Colors.GREEN}✓{Colors.END}")
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
    print(f"\n{Colors.YELLOW}Шаг 2: Выбор режима парсинга{Colors.END}")
    print("-" * 40)
    print("1) Парсить последние посты сабреддита")
    print("2) Парсить конкретный пост по ссылке")
    
    parse_mode = input("Выберите режим (1/2): ").strip()

    # Шаг 3: Ввод параметров
    print(f"\n{Colors.YELLOW}Шаг 3: Параметры парсинга{Colors.END}")
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
        print(f"{Colors.GREEN}✓{Colors.END} SentimentAnalyzer готов (модель: {model})")

    # Запуск парсинга
    print(f"\n{Colors.GREEN}{'='*60}")
    print("Начинаем парсинг...")
    print(f"{'='*60}{Colors.END}\n")

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
        print(f"\n{Colors.GREEN}{'='*60}")
        print(f"Парсинг завершён! Собрано {len(df)} уникальных пользователей")
        print(f"{'='*60}{Colors.END}")
        
        stats = parser.get_stats()
        print("\n--- Статистика ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"\n{Colors.RED}Данные не собраны{Colors.END}")

    input("\nНажмите Enter для продолжения...")


def run_web_interface():
    """Запускает Web интерфейс"""
    print(f"\n{Colors.PURPLE}{'='*60}")
    print("Запуск Web интерфейса...")
    print(f"{'='*60}{Colors.END}\n")

    # Проверяем Node.js с shell=True для Windows
    try:
        result = subprocess.run(
            "node --version",
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode != 0:
            raise FileNotFoundError("Node.js not found")
        node_version = result.stdout.strip()
        print(f"{Colors.GREEN}✓{Colors.END} Node.js версия: {node_version}")
    except FileNotFoundError:
        print(f"{Colors.RED}Ошибка: Node.js не найден{Colors.END}")
        print("Пожалуйста, установите Node.js с сайта https://nodejs.org/")
        input("Нажмите Enter для выхода...")
        return

    # Проверяем директорию сервера
    web_server_dir = Path(__file__).parent / "web-server"
    
    # Установка зависимостей Node.js если нужно
    print("\nПроверка зависимостей Node.js...")
    if not (web_server_dir / "node_modules").exists():
        print("Установка зависимостей Node.js...")
        try:
            subprocess.run(
                "npm install",
                cwd=str(web_server_dir),
                check=True,
                capture_output=True,
                shell=True
            )
            print(f"{Colors.GREEN}✓{Colors.END} Зависимости установлены")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}Ошибка установки зависимостей:{Colors.END}")
            print(e.stderr.decode() if e.stderr else str(e))
            input("Нажмите Enter для выхода...")
            return
        except FileNotFoundError:
            print(f"{Colors.RED}Ошибка: npm не найден{Colors.END}")
            input("Нажмите Enter для выхода...")
            return

    # Запуск Node.js сервера
    print("\nЗапуск веб-сервера...")
    print(f"{Colors.YELLOW}Сервер запускается на http://localhost:3000{Colors.END}")
    
    try:
        server_process = subprocess.Popen(
            "node server.js",
            cwd=str(web_server_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Ждём запуска сервера
        time.sleep(2)
        
        # Открываем браузер
        print(f"\n{Colors.GREEN}Открытие браузера...{Colors.END}")
        webbrowser.open("http://localhost:3000")
        
        print(f"\n{Colors.CYAN}{'='*60}")
        print("Web интерфейс запущен!")
        print(f"Откройте в браузере: http://localhost:3000")
        print(f"{'='*60}{Colors.END}")
        print(f"\n{Colors.YELLOW}Нажмите Ctrl+C для остановки сервера{Colors.END}")
        
        # Ждём пока сервер работает
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print(f"\n{Colors.RED}Остановка сервера...{Colors.END}")
            server_process.terminate()
            server_process.wait()
            
    except Exception as e:
        print(f"{Colors.RED}Ошибка запуска сервера:{Colors.END} {e}")
        input("Нажмите Enter для выхода...")


def main():
    """Главная функция"""
    while True:
        print_header()
        print_menu()
        
        choice = input(f"\n{Colors.BOLD}Ваш выбор:{Colors.END} ").strip()
        
        if choice == "1":
            run_cli_interface()
        elif choice == "2":
            run_web_interface()
            input("\nНажмите Enter для возврата в меню...")
        elif choice == "3" or choice.lower() in ['exit', 'quit', 'выход', 'q']:
            print(f"\n{Colors.CYAN}До свидания!{Colors.END}\n")
            break
        else:
            print(f"\n{Colors.RED}Неверный выбор. Попробуйте снова.{Colors.END}")
            time.sleep(1)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

