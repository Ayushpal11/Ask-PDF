import React from 'react';

export default function Header({ pdfName, handleFileUpload, handleReset }) {
    return (
        <header className="shadow-lg flex items-center justify-between h-20 min-h-[80px] px-6 sm:px-12">
            <img
                src="/AI_Planet_Logo.png"
                alt="AI Planet Logo"
                className="h-8 sm:h-12 cursor-pointer"
                onClick={handleReset} // Reset on clicking logo
            />
            <div className="flex items-center gap-4 h-full">
                {pdfName && (
                    <div className="flex items-center gap-2">
                        <img src="/pdf.svg" alt="PDF icon" />
                        <span
                            className="text-sm sm:text-base font-medium text-green-600 truncate max-w-[150px] sm:max-w-[200px] md:max-w-[300px]"
                            title={pdfName}
                        >
                            {pdfName}
                        </span>
                    </div>
                )}
                <label className="flex items-center justify-center h-10 w-10 sm:h-1/2 sm:w-48 border border-black rounded-lg font-inter font-semibold gap-2 sm:gap-5 hover:bg-gray-100 transition-colors duration-200 cursor-pointer">
                    <input
                        type="file"
                        accept="application/pdf"
                        className="hidden"
                        onChange={handleFileUpload} //Button for File Upload
                    />
                    <img src="/gala_add.svg" alt="Add icon" className="h-5" />
                    <span className="hidden sm:block">Upload PDF</span>
                </label>
            </div>
        </header>
    );
}
