const { useState, useRef, useEffect, useCallback } = React;

const SeriesMenu = ({ seriesList, onOpenViewer, isMenuOpen, onToggleMenu }) => {
    // Используем CSS-классы из вашего CSS файла
    if (!isMenuOpen) {
        return (
            <button
                className="menu-toggle"
                onClick={onToggleMenu}
                title="Показать список серий"
            >
                
            </button>
        );
    }

    return (
        <>
            <div 
                className="menu-overlay active"
                onClick={onToggleMenu}
            />
            <div className="series-list active">
                <div className="series-list-header">
                    <h3 className="series-list-title">Серии</h3>
                    <button
                        className="close-menu"
                        onClick={onToggleMenu}
                    >
                        ×
                    </button>
                </div>
                
                <div className="series-content">
                    <div className="series-count">
                        {seriesList.length > 0 
                            ? ' ' 
                            : 'Загрузите архив для просмотра серий'
                        }
                    </div>
                    
                    {seriesList.length > 0 && (
                        <>
                        <div className="export-section">
                                <SeriesInfoBtn 
                                    seriesList={seriesList}
                                />
                            </div>
                        <div className="series-grid">
                            {seriesList.map((series, index) => (
                                <button
                                    key={series.key}
                                    className="series-button"
                                    onClick={() => {
                                        onOpenViewer(series);
                                        onToggleMenu();
                                    }}
                                >
                                    {series.name}
                                </button>
                            ))}
                        </div>
                        </>
                    )}
                </div>
            </div>
        </>
    );
};