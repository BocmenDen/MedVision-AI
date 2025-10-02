const { useState, useRef, useEffect } = React;

const ZipUploader = ({
    onZipLoad,
    onUploadError,
    onHandleFile,
    isLoading = false
}) => {
    const [dragActive, setDragActive] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [showError, setShowError] = useState(false);
    const fileInputRef = useRef(null);

    const isAnyLoading = isLoading || isProcessing;

    useEffect(() => {
        if (errorMessage) {
            setShowError(true);
        } else {
            setShowError(false);
        }
    }, [errorMessage]);

    const processZipOnClient = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = function (e) {
                const zipData = e.target.result;
                JSZip.loadAsync(zipData)
                    .then(async (zip) => {
                        const dicomFiles = Object.keys(zip.files).filter(filename => filename.toLowerCase().endsWith('.dcm'));

                        if (dicomFiles.length === 0) {
                            const msg = 'ZIP архив не содержит DICOM файлов';
                            setErrorMessage(msg);
                            onUploadError?.(msg);
                            reject(new Error(msg));
                            return;
                        }

                        if (onZipLoad) {
                            onZipLoad(zip, file);
                        }
                        resolve(zip);
                    })
                    .catch(error => {
                        const msg = 'Ошибка чтения ZIP архива';
                        setErrorMessage(msg);
                        onUploadError?.(msg);
                        reject(error);
                    });
            };

            reader.onerror = () => {
                const msg = 'Ошибка чтения файла';
                setErrorMessage(msg);
                onUploadError?.(msg);
                reject(new Error('File reading error'));
            };

            reader.readAsArrayBuffer(file);
        });
    };

    const handleFile = async (file) => {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.zip')) {
            const msg = 'Пожалуйста, выберите ZIP файл';
            setErrorMessage(msg);
            onUploadError?.(msg);
            return;
        }

        if (file.size > 100 * 1024 * 1024) {
            const msg = 'Файл слишком большой. Максимальный размер: 100MB';
            setErrorMessage(msg);
            onUploadError?.(msg);
            return;
        }

        try {
            setErrorMessage('');
            setIsProcessing(true);
            await processZipOnClient(file);
        } catch (error) {
            console.error('File processing failed:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) {
            handleFile(file);
        }
        event.target.value = '';
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();

        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    };

    const handleClick = () => {
        if (!isAnyLoading) {
            fileInputRef.current?.click();
        }
    };

    return (
        <div className="zip-uploader" style={{ position: 'relative', paddingBottom: '48px' }}>
            <div
                className={`upload-area ${dragActive ? 'drag-active' : ''} ${isAnyLoading ? 'loading' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={handleClick}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".zip"
                    onChange={handleFileSelect}
                    disabled={isAnyLoading}
                    style={{ display: 'none' }}
                />

                <div className="upload-content" style={{ color: '#495057' }}>
                    {isAnyLoading ? (
                        <>
                            <div className="spinner" style={{
                                margin: '0 auto 10px',
                                width: '24px',
                                height: '24px',
                                border: '3px solid #842029',
                                borderTop: '3px solid transparent',
                                borderRadius: '50%',
                                animation: 'spin 1s linear infinite'
                            }}></div>
                            <span>Обработка архива...</span>
                        </>
                    ) : (
                        <>
                            <div style={{ fontWeight: 'bold', fontSize: '16px' }}>
                                Загрузите ZIP архив с DICOM файлами
                            </div>
                            <div style={{ fontSize: '14px', marginTop: '6px', color: '#6c757d' }}>
                                Нажмите или перетащите файл сюда
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Ошибка */}
            {showError && (
                <div
                    className="error-banner"
                    role="alert"
                    style={{
                        position: 'absolute',
                        bottom: '0',
                        left: '50%',
                        transform: 'translate(-50%, -8px)',
                        backgroundColor: '#f8d7da',
                        color: '#842029',
                        border: '1px solid #f5c2c7',
                        padding: '12px 20px',
                        borderRadius: '8px',
                        fontSize: '14px',
                        fontWeight: '500',
                        maxWidth: '90%',
                        boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                        userSelect: 'none',
                        zIndex: 100,
                        textAlign: 'left',
                        lineHeight: '1.4',
                        position: 'relative'
                    }}
                >
                    <span>{errorMessage}</span>
                    <button
                        onClick={() => setErrorMessage('')}
                        aria-label="Закрыть сообщение"
                        style={{
                            position: 'absolute',
                            top: '6px',
                            right: '10px',
                            background: 'transparent',
                            border: 'none',
                            fontSize: '18px',
                            fontWeight: 'bold',
                            color: '#842029',
                            cursor: 'pointer',
                            lineHeight: '1',
                        }}
                    >
                        ×
                    </button>
                </div>
            )}


            <style>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
};
