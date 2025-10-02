const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Создаем папку для загрузок если её нет
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// ВАЖНОООО НУЖНО ДОБАВИТЬ
app.use('/uploads', express.static(uploadsDir));

// Настройка multer для загрузки файлов
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
        // Генерируем уникальное имя файла
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, 'dicom-' + uniqueSuffix + '.zip');
    }
});

// Фильтр для проверки типа файла
const fileFilter = (req, file, cb) => {
    if (file.mimetype === 'application/zip' || 
        file.mimetype === 'application/x-zip-compressed' ||
        path.extname(file.originalname).toLowerCase() === '.zip') {
        cb(null, true);
    } else {
        cb(new Error('Разрешены только ZIP файлы'), false);
    }
};

const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: {
        fileSize: 100 * 1024 * 1024
    }
});

// Маршрут для загрузки ZIP файла
app.post('/api/upload-zip', upload.single('zipFile'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'Файл не был загружен'
            });
        }

        console.log('Файл загружен:', req.file.filename);
        console.log('Полный путь:', req.file.path);
        console.log('Размер:', req.file.size, 'bytes');
        
        // Проверяем, что файл действительно существует
        if (!fs.existsSync(req.file.path)) {
            return res.status(500).json({
                success: false,
                message: 'Файл не был сохранен на сервере'
            });
        }

        res.json({
            success: true,
            message: 'Файл успешно загружен',
            file: {
                filename: req.file.filename,
                originalName: req.file.originalname,
                size: req.file.size,
                path: req.file.path,
                uploadDate: new Date().toISOString(),
                // Добавляем полный URL для скачивания
                downloadUrl: `http://localhost:${PORT}/uploads/${req.file.filename}`
            }
        });

    } catch (error) {
        console.error('Ошибка загрузки:', error);
        res.status(500).json({
            success: false,
            message: 'Ошибка при загрузке файла: ' + error.message
        });
    }
});

// Маршрут для получения списка файлов (для отладки)
app.get('/api/files', (req, res) => {
    try {
        const files = fs.readdirSync(uploadsDir);
        const fileDetails = files.map(filename => {
            const filePath = path.join(uploadsDir, filename);
            const stats = fs.statSync(filePath);
            return {
                filename,
                size: stats.size,
                created: stats.birthtime,
                url: `http://localhost:${PORT}/uploads/${filename}`
            };
        });
        
        res.json({
            success: true,
            files: fileDetails
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Ошибка при чтении файлов: ' + error.message
        });
    }
});

// Маршрут для проверки здоровья сервера
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        timestamp: new Date().toISOString(),
        service: 'DICOM ZIP Uploader',
        uploadsDirectory: uploadsDir,
        uploadsExists: fs.existsSync(uploadsDir)
    });
});

// Обработка ошибок multer
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                success: false,
                message: 'Файл слишком большой. Максимальный размер: 100MB'
            });
        }
    }
    
    res.status(400).json({
        success: false,
        message: error.message
    });
});

// Запуск сервера
app.listen(PORT, () => {
    console.log(`Сервер запущен на порту ${PORT}`);
    console.log(`API загрузки: http://localhost:${PORT}/api/upload-zip`);
    console.log(`Статические файлы: http://localhost:${PORT}/uploads/`);
    console.log(`Список файлов: http://localhost:${PORT}/api/files`);
    console.log(`Health check: http://localhost:${PORT}/api/health`);
    console.log(`Папка загрузок: ${uploadsDir}`);
    
    // Проверяем права доступа к папке
    try {
        const testFile = path.join(uploadsDir, 'test.txt');
        fs.writeFileSync(testFile, 'test');
        fs.unlinkSync(testFile);
        console.log('✅ Права на запись в папку uploads: OK');
    } catch (error) {
        console.error('❌ Ошибка прав доступа к папке uploads:', error.message);
    }
});