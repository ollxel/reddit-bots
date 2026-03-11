const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Store for parsing results
let currentResults = null;
let parsingStatus = {
    status: 'idle',
    message: '',
    progress: 0
};

// Endpoint to start parsing
app.post('/api/parse', (req, res) => {
    const { apiKey, model, parseMode, subreddit, postUrl, targetComments } = req.body;
    
    console.log('Received parse request:', { parseMode, subreddit, postUrl });
    
    parsingStatus = {
        status: 'starting',
        message: 'Подготовка к парсингу...',
        progress: 5
    };
    currentResults = null;
    
    // Python script path
    const pythonScript = path.join(__dirname, '..', 'web_parser.py');
    
    // Arguments for the Python script
    const args = [
        pythonScript,
        '--api-key', apiKey || '',
        '--model', model || 'arcee-ai/trinity-large-preview:free',
        '--mode', parseMode,
        '--subreddit', subreddit || '',
        '--post-url', postUrl || '',
        '--target', targetComments || '100'
    ];
    
    console.log('Spawning Python with args:', args);
    
    const pythonProcess = spawn('python', args, {
        cwd: path.join(__dirname, '..'),
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        console.log('Python stdout:', output);
        outputData += output;
        
        // Parse progress from output
        if (output.includes('Отправляем POST-запросы')) {
            parsingStatus.message = 'Отправляем POST-запросы...';
            parsingStatus.progress = 20;
        } else if (output.includes('Парсим комментарии')) {
            parsingStatus.message = 'Парсим комментарии...';
            parsingStatus.progress = 40;
        } else if (output.includes('Анализируем тональность')) {
            parsingStatus.message = 'Анализируем тональность...';
            parsingStatus.progress = 60;
        } else if (output.includes('Сохранение данных')) {
            parsingStatus.message = 'Сохранение данных...';
            parsingStatus.progress = 80;
        } else if (output.includes('Parsing complete') || output.includes('собрано')) {
            parsingStatus.message = 'Парсинг завершён!';
            parsingStatus.progress = 100;
            parsingStatus.status = 'completed';
        }
    });
    
    pythonProcess.stderr.on('data', (data) => {
        const error = data.toString();
        console.error('Python stderr:', error);
        errorData += error;
    });
    
    pythonProcess.on('close', (code) => {
        console.log('Python process exited with code:', code);
        
        if (code === 0) {
            parsingStatus.status = 'completed';
            parsingStatus.message = 'Парсинг завершён!';
            parsingStatus.progress = 100;
            
            // Try to read the results
            const resultsPath = path.join(__dirname, '..', 'web_results.json');
            if (fs.existsSync(resultsPath)) {
                try {
                    currentResults = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
                } catch (e) {
                    console.error('Error parsing results:', e);
                }
            }
        } else {
            parsingStatus.status = 'error';
            parsingStatus.message = 'Ошибка парсинга: ' + errorData.slice(-200);
        }
    });
    
    res.json({ success: true, message: 'Парсинг начат' });
});

// Endpoint to get parsing status
app.get('/api/status', (req, res) => {
    res.json(parsingStatus);
});

// Endpoint to get results
app.get('/api/results', (req, res) => {
    if (currentResults) {
        res.json(currentResults);
    } else {
        res.json({ error: 'Нет результатов' });
    }
});

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`\n🌐 Web интерфейс запущен: http://localhost:${PORT}\n`);
});

