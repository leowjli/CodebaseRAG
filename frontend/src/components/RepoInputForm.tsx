import { useState } from 'react';
import axios from '../api';

interface RepoInputFormProps {
    setRepoPath: React.Dispatch<React.SetStateAction<string>>;
}

function RepoInputForm({ setRepoPath }: RepoInputFormProps) {
    const [repoUrl, setRepoUrl] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);

    const handleRepoUrlChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
        setRepoUrl(e.target.value);
    };

    const cloneRepo = async (): Promise<void> => {
        if (!repoUrl) return;
        setLoading(true);
        try {
            const response = await axios.post("http://localhost:8000/clone-repository/", { repo_url: repoUrl });
            setRepoPath(response.data.message);
            setLoading(false);
        } catch(err) {
            console.error("Error fetching Repository", err);
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Enter a Repository</h2>
            <input
                type="text"
                value={repoUrl}
                onChange={handleRepoUrlChange}
                placeholder="Enter GitHub repository URL"
            />
            <button onClick={cloneRepo} disabled={loading}>
                {loading ? "Cloning..." : "Clone Repository"}
            </button>
        </div>
    );
};

export default RepoInputForm;