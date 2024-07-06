import React, { FC, useState, useEffect } from 'react';
import { IconReload, IconCopy } from "@tabler/icons-react";

interface AnswerProps {
    searchQuery: string;
    answer: string;
    done: boolean;
    onReset: () => void;
}

export const Answer: FC<AnswerProps> = ({ searchQuery, answer, done, onReset }) => {
    const answerText = getValueFromJsonByKey(answer, 'result');
    const sourcesText = getValueFromJsonByKey(answer, 'retrieval');
    const sourcesDict = parseRetrievalsEnhance(sourcesText);
    const question = getValueFromJsonByKey(answer, 'query');

    const [selectedKey, setSelectedKey] = useState<string | null>(null);
    const [expanded, setExpanded] = useState<boolean>(true); // Set to true to expand by default
    const [copied, setCopied] = useState<string | null>(null);

    useEffect(() => {
        if (selectedKey !== null) {
            setCopied(null); // Reset copied state when a new source is selected
        }
    }, [selectedKey]);

    const handleKeyClick = (key: string) => {
        setSelectedKey(prevKey => prevKey === key ? null : key); // Toggle selected key
    };

    const handleExpandToggle = () => {
        setExpanded(!expanded);
    };

    const handleCopy = (text: string, type: string) => {
        navigator.clipboard.writeText(text)
            .then(() => setCopied(type))
            .catch((error) => console.error('Failed to copy:', error));
    };

    const handleDoubleClick = (key: string) => {
        setSelectedKey(prevKey => prevKey === key ? null : key); // Toggle selected key
    };

    const handlePart2Click = async (url: string) => {
        url = url.trim();
        console.log("Clicked URL:", url);
        try {
            window.open(url, '_blank', 'noopener,noreferrer');
        } catch (error) {
            console.error('Failed to open URL:', url, error);
        }
    };

    const renderPart2 = (part2: string) => {
        const httpIndex = part2.toLowerCase().indexOf('http');
        if (httpIndex !== -1) {
            const beforeUrl = part2.substring(0, httpIndex);
            const url = part2.substring(httpIndex).trim();
            return (
                <span>
                    {beforeUrl}
                    <span
                        onClick={() => handlePart2Click(url)}
                        style={{ cursor: 'pointer', color: 'green' }}
                    >
                        {url}
                    </span>
                </span>
            );
        } else {
            return <span>{part2}</span>;
        }
    };

    return (
        <div className="flex flex-col min-h-screen">
            <div className="text-center text-4xl font-bold mb-6">
                {question}
            </div>
            <div className="flex-grow flex justify-center items-start space-x-6 mx-auto p-6 bg-background text-text rounded-lg shadow-lg max-w-[1400px]">
                <div className="w-1/2 flex flex-col items-center justify-center space-y-6">
                    <div className="w-full border-b border-gray-600 pb-4 mb-4">
                        <div className="flex justify-between items-center mb-2">
                            <div className="text-lg text-blue-400">Answer</div>
                            <button className="text-blue-400 hover:underline" onClick={() => handleCopy(answerText, 'answer')}>
                                {copied === 'answer' ? 'Copied!' : 'Copy Answer'}
                            </button>
                        </div>
                        <div className="mt-2 text-lg leading-relaxed text-center">{answerText}</div>
                    </div>

                    {done && (
                        <button
                            className="flex items-center justify-center mt-8 h-12 w-48 rounded-full bg-blue-500 text-white hover:bg-blue-600 transition-all duration-300"
                            onClick={onReset}
                        >
                            <IconReload size={20} />
                            <div className="ml-2">Ask New Question</div>
                        </button>
                    )}
                </div>

                <div className="w-1/2 flex flex-col items-center space-y-6">
                    {done && (
                        <>
                            <div className="w-full border-b border-gray-600 pb-4 mb-4">
                                <div className="flex justify-between items-center cursor-pointer" onClick={handleExpandToggle}>
                                    <div className="text-lg text-blue-400">View Selected Retrievals</div>
                                    <IconReload size={20} className="text-blue-400" />
                                </div>
                                {expanded && (
                                    <div className="mt-4 grid gap-4 max-w-full">
                                        <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                                            {Object.entries(sourcesDict).map(([key, value], index) => (
                                                <button
                                                    key={index}
                                                    title={value.part2}
                                                    className="tooltip-btn text-lg rounded-lg shadow-md bg-gray-800 text-white hover:bg-gray-900 hover:shadow-lg flex items-center justify-center p-2 transition-all duration-300"
                                                    onClick={() => handleKeyClick(key)}
                                                    onDoubleClick={() => handleDoubleClick(key)}
                                                >
                                                    {key}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {selectedKey && (
                                <div className="w-full border border-gray-600 mt-4 p-4 rounded-lg">
                                    <div className="flex justify-between items-center mb-2">
                                        <div className="text-lg text-blue-400">Selected Source</div>
                                        <button className="text-blue-400 hover:underline" onClick={() => handleCopy(sourcesDict[selectedKey].part2, 'source')}>
                                            {copied === 'source' ? 'Copied!' : 'Copy'}
                                        </button>
                                    </div>
                                    <div className="leading-relaxed">
                                        {sourcesDict[selectedKey].part1}
                                        <br />
                                        <strong>
                                            {renderPart2(sourcesDict[selectedKey].part2)}
                                        </strong>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
            <div className="flex-shrink-0 p-4 bg-background text-text text-center">
            </div>
        </div>
    );
};

const getValueFromJsonByKey = (jsonString: string, key: string): string => {
    try {
        const dict = JSON.parse(jsonString);
        if (key in dict) {
            return dict[key];
        } else {
            console.error("Key not found in dictionary.");
            return 'Key not found';
        }
    } catch (error) {
        console.error("Failed to parse JSON:", error);
        return 'Invalid JSON';
    }
};

const parseRetrievals = (inputText: string): Record<string, string> => {
    const pattern = /Retrieval (\d+):([\s\S]*?)(?=Retrieval \d+:|$)/g;
    let match;
    const result: Record<string, string> = {};

    while ((match = pattern.exec(inputText)) !== null) {
        const key = `Retrieval ${match[1]}`;
        const value = match[2].trim();
        result[key] = value;
    }

    return result;
};

const parseRetrievalsEnhance = (inputText: string): Record<string, { part1: string, part2: string }> => {
    const pattern = /Retrieval (\d+):([\s\S]*?)(?=Retrieval \d+:|$)/g;
    let match;
    const result: Record<string, { part1: string, part2: string }> = {};

    while ((match = pattern.exec(inputText)) !== null) {
        const key = `Retrieval ${match[1]}`;
        const splitIndex = match[2].indexOf("This retrieval is performed from the document");
        const part1 = match[2].substring(0, splitIndex).trim();
        const part2 = match[2].substring(splitIndex).trim();
        result[key] = { part1, part2 };
    }

    return result;
};
