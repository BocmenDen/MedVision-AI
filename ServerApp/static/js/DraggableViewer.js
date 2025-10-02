const { useState, useRef, useEffect, useCallback } = React;

const DraggableViewer = ({ viewer, onClose, onPositionUpdate, onFocus }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
    const windowRef = useRef(null);

    const handleMouseDown = (e) => {
        if (e.target.closest('.window-content') || e.target.tagName === 'BUTTON') return;

        setIsDragging(true);
        setDragOffset({
            x: e.clientX - viewer.position.x,
            y: e.clientY - viewer.position.y
        });
        onFocus();
        e.preventDefault();
    };

    const handleMouseMove = useCallback((e) => {
        if (!isDragging) return;

        const newX = e.clientX - dragOffset.x;
        const newY = e.clientY - dragOffset.y;

        const maxX = window.innerWidth - 400;
        const maxY = window.innerHeight - 500;

        onPositionUpdate({
            x: Math.max(0, Math.min(newX, maxX)),
            y: Math.max(0, Math.min(newY, maxY))
        });
    }, [isDragging, dragOffset, onPositionUpdate]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
    }, []);

    useEffect(() => {
        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging, handleMouseMove, handleMouseUp]);

    if (!viewer.isOpen) return null;

    return (
        <div
            ref={windowRef}
            style={{
                position: 'fixed',
                left: `${viewer.position.x}px`,
                top: `${viewer.position.y}px`,
                width: '400px',
                height: '500px',
                background: 'white',
                border: '2px solid #3f3f3fff',
                borderRadius: '10px',
                boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
                zIndex: viewer.zIndex,
                cursor: isDragging ? 'grabbing' : 'default',
                display: 'flex',
                flexDirection: 'column',
                resize: 'both',
                overflow: 'hidden',
                minWidth: '300px',
                minHeight: '400px'
            }}
            onMouseDown={handleMouseDown}
        >
            <div style={{
                background: '#e1d5b5',
                color: '#312F2c',
                padding: '10px 15px',
                borderTopLeftRadius: '8px',
                borderTopRightRadius: '8px',
                cursor: 'move',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                userSelect: 'none',
                flexShrink: 0
            }}>
                <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
                    {viewer.series.name}
                </span>
                <button
                    onClick={onClose}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: 'white',
                        fontSize: '18px',
                        cursor: 'pointer',
                        padding: '0',
                        width: '20px',
                        height: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}
                    title="Закрыть окно"
                >
                    ×
                </button>
            </div>

            <div className="window-content" style={{
                flex: 1,
                padding: '10px',
                overflow: 'auto',
                minHeight: 0
            }}>
                <SeriesViewer
                    files={viewer.series.files}
                    seriesName={viewer.series.name}
                    model1={viewer.series.model1}
                    compact={true}
                />
            </div>
        </div>
    );
};
