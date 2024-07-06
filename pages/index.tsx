import { Answer } from "@/components/Answer";
import { Search } from "@/components/Search";
import { SearchQuery } from "@/types";
import Head from "next/head";
import { useState, useEffect } from "react";
import NetopLogo from "@/components/NetopLogo";
import { IconCellSignal5 } from '@tabler/icons-react';
import { IconSun, IconMoon } from "@tabler/icons-react"; 

export default function Home() {
  const [searchQuery, setSearchQuery] = useState<SearchQuery>({ query: "", sourceLinks: [] });
  const [answer, setAnswer] = useState<string>("");
  const [done, setDone] = useState<boolean>(false);
  const [theme, setTheme] = useState<string>("light");

  const handleNetopClick = () => {
    window.open("https://github.com/netop-team", "_blank");
  };

  const handleDisclaimerClick = () => {
    window.open("https://arxiv.org/pdf/2404.15939v1.pdf", "_blank");
  };

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === "dark" ? "light" : "dark"));
  };

  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  return (
    <>
      <Head>
        <title>Telco-RAG</title>
        <meta name="description" content="AI-powered search." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/Netop.png" />
      </Head>

      <div className="flex flex-col h-screen overflow-hidden bg-background text-text">
        <div className="flex justify-end p-4">
          <button
            className="flex items-center p-2 rounded-full focus:outline-none"
            onClick={toggleTheme}
            aria-label="Toggle Theme"
          >
            {theme === "dark" ? <IconSun size={24} /> : <IconMoon size={24} />}
          </button>
        </div>

        <div className="flex-grow overflow-auto">
          {answer ? (
            <Answer
              searchQuery={searchQuery}
              answer={answer}
              done={done}
              onReset={() => {
                setAnswer("");
                setSearchQuery({ query: "", sourceLinks: [] });
                setDone(false);
              }}
            />
          ) : (
            <Search
              onSearch={setSearchQuery}
              onAnswerUpdate={(value) => setAnswer((prev) => prev + value)}
              onDone={setDone}
            />
          )}
        </div>

        <footer className="flex-shrink-0 p-4 bg-background text-center">
          <a href="#" onClick={handleNetopClick}>
            <NetopLogo />
          </a>
          <p className="text-sm">
            Telco-RAG is an under-development research project and it is designed to answer technical questions related to 3GPP Standards. More details
            <a href="#" onClick={handleDisclaimerClick} className="text-link underline ml-1">here</a>.
          </p>
        </footer>
      </div>
    </>
  );
}
