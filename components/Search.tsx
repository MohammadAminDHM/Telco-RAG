import React, { FC, KeyboardEvent, useEffect, useRef, useState } from "react";
import { IconArrowRight, IconFileSignal, IconSearch } from "@tabler/icons-react";

interface SearchProps {
  onSearch: (searchResult: any) => void;
  onAnswerUpdate: (answer: string) => void;
  onDone: (done: boolean) => void;
}

const modelDescriptions = {
  "gpt-3.5": "GPT-3.5: Performed <score> accuracy on TeleQnA.",
  "gpt-4": "GPT-4: Performed <score> accuracy on TeleQnA.",
  "gpt-4o": "GPT-4o: Performed <score> accuracy on TeleQnA.",
};

const modelPlaceholders = {
  "gpt-3.5": "Enter your OpenAI key for GPT-3.5",
  "gpt-4": "Enter your OpenAI key for GPT-4",
  "gpt-4o": "Enter your OpenAI key for GPT-4o",
};

export const Search: FC<SearchProps> = ({ onSearch, onAnswerUpdate, onDone }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-3.5");
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!query || !apiKey) {
      setError("Please enter a valid query and API key.");
      return;
    }
    setLoading(true);
    try {
      await handleStream(query, model, apiKey);
    } catch (error) {
      console.error("Failed to send the query:", error);
      setLoading(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  useEffect(() => {
    const storedApiKey = localStorage.getItem("OPENAI_KEY");
    if (storedApiKey) {
      setApiKey(storedApiKey);
    }
    const storedModel = localStorage.getItem("SELECTED_MODEL");
    if (storedModel) {
      setModel(storedModel);
    }
    inputRef.current?.focus();
  }, []);

  const handleToggleSettings = () => {
    setShowSettings(!showSettings);
  };

  const handleStream = async (query: string, model: string, apiKey: string) => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/process_query/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, model_name: model, api_key: apiKey }),
      });

      if (!response.ok) {
        throw new Error(`HTTP status ${response.status}`);
      }

      const responseData = await response.json();
      if (responseData) {
        const answer = responseData;
        onAnswerUpdate(answer);
      }

      setLoading(false);
      onDone(true);
    } catch (error) {
      console.error("Failed to process the query:", error);
      onAnswerUpdate("Error in processing query");
      setLoading(false);
    }
  };

  const handleSave = () => {
    localStorage.setItem("OPENAI_KEY", apiKey);
    localStorage.setItem("SELECTED_MODEL", model);
    setShowSettings(false);
    inputRef.current?.focus();
  };

  const handleClear = () => {
    localStorage.removeItem("OPENAI_KEY");
    localStorage.removeItem("SELECTED_MODEL");
    setApiKey("");
    setModel("gpt-3.5");
  };

  return (
    <div className="mx-auto w-full max-w-[750px] flex flex-col items-center space-y-4 px-3 pt-16 sm:pt-32">
      <div className="flex items-center mb-4">
        <IconFileSignal size={50} />
        <div className="ml-2 text-center text-4xl font-semibold">Telco-RAG</div>
      </div>

      <div className="relative w-full">
        <IconSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 h-6 w-6 text-[#D4D4D8] opacity-50" />
        <input
          ref={inputRef}
          className="h-12 w-full rounded-full border border-zinc-600 bg-[#20232A] px-16 py-2 text-xl text-white focus:border-zinc-800 focus:bg-[#282C34] focus:outline-none focus:ring-2 focus:ring-zinc-800 transition-all duration-300"
          type="text"
          placeholder="Ask anything..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className={`absolute right-4 top-1/2 transform -translate-y-1/2 h-8 w-8 rounded-full bg-blue-500 p-1 hover:bg-blue-600 transition-all duration-300 ${loading ? "cursor-not-allowed" : "cursor-pointer"}`}
          onClick={handleSearch}
          disabled={loading}
        >
          {loading ? (
            <div className="h-full w-full animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></div>
          ) : (
            <IconArrowRight />
          )}
        </button>
      </div>

      <button
        className="flex cursor-pointer items-center space-x-2 rounded-full border border-zinc-600 px-4 py-2 text-sm text-[#A9A9A9] hover:text-white transition-all duration-300"
        onClick={handleToggleSettings}
      >
        {showSettings ? "Hide" : "Show"} Settings
      </button>

      {showSettings && (
        <div className="w-full max-w-[400px] p-4 rounded-lg border border-zinc-600 bg-[#2A2A31] shadow-lg space-y-4 transition-all duration-300">
          <select
            className="w-full rounded-md border border-gray-300 p-2 text-lg text-black shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="gpt-3.5">gpt-3.5</option>
            <option value="gpt-4">gpt-4</option>
            <option value="gpt-4o">gpt-4o</option>
            {/* Add more models as needed */}
          </select>

          <p className="text-center text-xs text-[#D4D4D8]">{modelDescriptions[model]}</p>

          <input
            type="password"
            className="w-full rounded-md border border-gray-300 p-2 text-lg text-black shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value);
              setError("");
            }}
            placeholder={modelPlaceholders[model]}
          />

          <div className="flex space-x-2">
            <button
              className="w-full rounded-full border border-zinc-600 bg-blue-500 px-4 py-2 text-sm text-white hover:bg-blue-600 transition-all duration-300"
              onClick={handleSave}
            >
              Save
            </button>
            <button
              className="w-full rounded-full border border-zinc-600 bg-red-500 px-4 py-2 text-sm text-white hover:bg-red-600 transition-all duration-300"
              onClick={handleClear}
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {error && <div className="text-red-500">{error}</div>}

      {loading && (
        <div className="flex items-center justify-center pt-8">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></div>
          <div className="mt-8 text-2xl">Getting answer...</div>
        </div>
      )}
    </div>
  );
};
