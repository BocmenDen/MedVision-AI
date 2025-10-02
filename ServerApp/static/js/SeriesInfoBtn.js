const { useState, useRef, useEffect, useCallback } = React;

/*Компонент для экспорта основной информации по всем сериям*/
const SeriesInfoBtn = ({ seriesList = [], disabled = false }) => {
    const [isExporting, setIsExporting] = useState(false);
    const [progress, setProgress] = useState(0);

    /*Создает CSV строку из данных*/
    const createCSV = (data) => {
        if (!data || data.length === 0) return '';

        const headers = [...new Set(data.flatMap(item => Object.keys(item)))];
        const csvHeaders = headers.join(';');

        const csvRows = data.map(item => {
            return headers.map(header => {
                const value = item[header] || '';
                if (typeof value === 'string' && (value.includes(';') || value.includes('"') || value.includes('\n'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            }).join(';');
        });

        return [csvHeaders, ...csvRows].join('\n');
    };

    /*Скачивает CSV файл*/
    const downloadCSV = (csvContent, filename) => {
        const BOM = '\uFEFF';
        const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');

        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        URL.revokeObjectURL(url);
    };

    /*Экспорт сводной информации по всем сериям*/
    const handleAllSeriesSummary = async () => {
        if (!seriesList || seriesList.length === 0) {
            alert('Нет загруженных серий для экспорта');
            return;
        }

        setIsExporting(true);
        setProgress(0);

        try {
            const summaryData = [];

            for (let i = 0; i < seriesList.length; i++) {
                const s = seriesList[i];
                const ai = s.model1 || {};
                const seriesSummary = {
                    'series_uid': s.name || 'Неизвестно',
                    'probability_of_pathology': ai.prob ? `${(ai.prob * 100).toFixed(2)}%` : 'Н/Д',
                    'pathology_class': ai.classIndex ?? 'Н/Д',
                    'time_of_processing': s.duration,
                    'Количество файлов': s.files?.length || 0,
                    'Наименование признака': ai.index_to_russian ?? 'Н/Д',
                    'Дата экспорта': new Date().toLocaleString('ru-RU')
                };

                summaryData.push(seriesSummary);

                setProgress(Math.round(((i + 1) / seriesList.length) * 100));
            }

            const csvContent = createCSV(summaryData);
            const filename = `сводка_серий_${new Date().toISOString().split('T')[0]}.csv`;
            downloadCSV(csvContent, filename);

            console.log(`Экспортировано ${summaryData.length} серий`);
        } catch (error) {
            console.error('Ошибка при экспорте:', error);
            alert('Произошла ошибка при создании сводки');
        } finally {
            setIsExporting(false);
            setTimeout(() => setProgress(0), 1000);
        }
    };

    const totalFiles = seriesList.reduce((sum, s) => sum + (s.files?.length || 0), 0);
    const isButtonDisabled = disabled || isExporting || seriesList.length === 0;
    const buttonBackground = seriesList.length === 0 ? '#6c757d' : '#842029';

    return (
        <div className="series-export-button">
            <div className="series-export-title">
                Экспорт сводки по сериям
            </div>

            <div className="series-export-stats">
                Загружено серий: {seriesList.length} | Всего файлов: {totalFiles}
            </div>

            {isExporting && (
                <div className="series-export-progress">
                    <div>Экспорт сводки: {progress}%</div>
                    <div className="series-export-progress-bar">
                        <div 
                            className="series-export-progress-fill"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                </div>
            )}

            <button
                onClick={handleAllSeriesSummary}
                disabled={isButtonDisabled}
                style={{ background: isExporting ? '#6c757d' : buttonBackground }}
                title={seriesList.length === 0 ?
                    "Нет загруженных серий для экспорта" :
                    "Экспорт сводной информации по всем сериям"
                }
            >
                {isExporting ? (
                    <>
                        <div className="series-export-spinner"></div>
                        Экспорт...
                    </>
                ) : (
                    <>
                        Экспорт сводки CSV
                    </>
                )}
            </button>
            {!isExporting && (
                <div className="series-export-info">
                    <div><strong>Будет экспортировано:</strong> {seriesList.length} серий</div>
                    <div className="series-export-info-small">
                        series_uid; probability_of_pathology ; pathology_class; time_of_processing; Количество файлов; Наименование признака; Дата экспорта
                    </div>
                </div>
            )}
        </div>
    );
};