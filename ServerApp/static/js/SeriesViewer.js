const { useState, useRef, useEffect, useCallback } = React;


const SeriesViewer = ({ files, seriesName, model1, compact = false }) => {
    const canvasRef = useRef(null);
    const [currentSlice, setCurrentSlice] = useState(0);
    const [windowCenter, setWindowCenter] = useState(400);
    const [isInverted, setIsInverted] = useState(false);
    const [windowWidth, setWindowWidth] = useState(800);
    const [isDragging, setIsDragging] = useState(false);
    const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

    // Загружаем срез при изменении currentSlice
    useEffect(() => {
        if (canvasRef.current && files.length > 0) {
            cornerstone.enable(canvasRef.current);
            loadSlice(currentSlice);
        }

        return () => {
            if (canvasRef.current) {
                cornerstone.disable(canvasRef.current);
            }
        };
    }, [files, currentSlice]);
    
    //загрузка среза
    const loadSlice = async (sliceIndex) => {
        if (!files[sliceIndex]) return;

        try {
            const dicomData = files[sliceIndex];
            const blob = new Blob([dicomData], { type: 'application/dicom' });
            const url = URL.createObjectURL(blob);

            const image = await cornerstone.loadImage(`wadouri:${url}`);
            await cornerstone.displayImage(canvasRef.current, image);

            if (sliceIndex === 0) {
                setWindowCenter(image.windowCenter || 400);
                setWindowWidth(image.windowWidth || 800);
            }

            updateViewport();
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error loading slice:', error);
        }
    };

    const updateViewport = () => {
        const element = canvasRef.current;
        if (!element) return;

        const viewport = cornerstone.getViewport(element);
        if (viewport) {
            viewport.voi = {
                windowCenter: windowCenter,
                windowWidth: windowWidth
            };
            viewport.invert = isInverted;
            
            cornerstone.setViewport(element, viewport);
        }
    };


    const changeSlice = (delta) => {
        setCurrentSlice(prevSlice => {
            const newSlice = prevSlice + delta;
            if (newSlice >= 0 && newSlice < files.length) {
                return newSlice;
            }
            return prevSlice;
        });
    };
    const toggleInversion = () => {
        setIsInverted(prev => !prev);
    };

    //обработчики для регулировки контрастности
    const handleMouseDown = (e) => {
        if (e.button === 0) {
            setIsDragging(true);
            setLastMousePos({ x: e.clientX, y: e.clientY });
        }
    };

    const handleMouseMove = (e) => {
        if (isDragging) {
            const deltaX = e.clientX - lastMousePos.x;
            const deltaY = e.clientY - lastMousePos.y;

            setWindowWidth(prev => Math.max(50, prev + deltaX * 2));
            setWindowCenter(prev => Math.max(1, prev + deltaY * 2));
            setLastMousePos({ x: e.clientX, y: e.clientY });
            updateViewport();
        }
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    //обработчик колеса мыши
    const handleWheel = (e) => {
        e.preventDefault();
        const delta = Math.sign(e.deltaY);
        changeSlice(delta);
    };

    //обработчики событий
    useEffect(() => {
        const element = canvasRef.current;
        if (!element) return;

        element.addEventListener('wheel', handleWheel);
        element.addEventListener('mousedown', handleMouseDown);

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);

        return () => {
            element.removeEventListener('wheel', handleWheel);
            element.removeEventListener('mousedown', handleMouseDown);
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging, lastMousePos, files.length]);

    // Обновляем viewport при изменении windowCenter/windowWidth
    useEffect(() => {
        updateViewport();
    }, [windowCenter, windowWidth, isInverted]);

    return (
    <div style={{
        width: '100%',
        height: compact ? '300px' : '512px',
        display: 'flex',
        flexDirection: 'column'
    }}>
        <div style={{
            padding: '5px',
            background: '#f0f0f0',
            fontSize: '12px',
            fontFamily: 'Courier New, monospace'
        }}>
            <span>
                Series: {seriesName} | Slice: {currentSlice + 1}/{files.length} | WC: {Math.round(windowCenter)} | WW: {Math.round(windowWidth)}
            </span>
        </div>

        <div
            ref={canvasRef}
            style={{
                flex: 1,
                background: '#000',
                cursor: isDragging ? 'grabbing' : 'grab',
                minHeight: compact ? '250px' : '400px'
            }}
            onContextMenu={(e) => e.preventDefault()}
        />
        <button
                onClick={toggleInversion}
                style={{
                    background: isInverted ? '#312F2c' : '#e1d5b5',
                    color: isInverted ? 'white' : '#312F2c',
                    margin: '2%',
                    border: 'none',
                    padding: '3px 8px',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    marginLeft: '10px'
                }}
                title={isInverted ? 'Отключить инверсию' : 'Включить инверсию'}
            >
                {isInverted ? 'Нормальный режим' : 'Инверсия'}
            </button>
            <div className="info-ai">
                {model1.index_to_russian} с вероятностью {model1.prob * 100}%
            </div>
            <div style={{ padding: '10px', background: '#f8f8f8', fontSize: '12px' }}>
                <p>• Колесо мыши - навигация по срезам</p>
                <p>• Зажать и потянуть - регулировка контрастности (горизонтально - WW, вертикально - WC)</p>
            </div>
    </div>
);
};
