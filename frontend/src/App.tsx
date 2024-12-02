import { useState } from 'react'
import RepoInputForm from './components/RepoInputForm';
import FunctionsForm from './components/FunctionForm';
import Chat from './components/Chat.tsx';
import './App.css'

function App() {
    const [repoPath, setRepoPath] = useState<string>('');
    const [funcs, setFuncs] =useState([]);

    return (
        <div className='App'>
            <h1>Codebase RAG chatbot</h1>

            <RepoInputForm setRepoPath={setRepoPath} />
            {repoPath && (
                <div>
                    <FunctionsForm repoPath={repoPath} setFunctions={setFuncs} />
                    {funcs.length > 0 && (
                        <div>
                            <h3>Functions in the codebase:</h3>
                            <ul>
                                {funcs.map((fn, index) => (
                                    <li key={index}>{fn}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
            <Chat />
        </div>
    );
}

export default App
