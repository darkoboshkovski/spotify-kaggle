import './App.css';
import Button from '@material-ui/core/Button';
import {useState} from "react";
import {CircularProgress, Grid, Typography} from "@material-ui/core";
import MaterialTable from "material-table";

function App() {

    const [appState, setAppState] = useState({data: [], loading: false});
    const [selectedPlaylist, setSelectedPlaylist] = useState(0);

    const onClick = () => {
        setAppState({...appState, loading: true});
        fetch("http://localhost:8000/playlists", {method: "GET"}).then(result => {
            result.json().then(json => {
                setAppState({loading: false, data: [...json]})
            })
        })
    }

    let componentToRender;
    if (appState.loading === false && appState.data.length === 0) {
        componentToRender = (<Button variant="contained" color="primary" onClick={onClick}>Get Playlists</Button>);
    } else if (appState.loading) {
        componentToRender = <CircularProgress/>
    } else {
        componentToRender =
            <Grid container direction="column">
                <Typography>
                    {`Features with lowest Standard Dev: ${appState.data[selectedPlaylist].feats.join(", ")}`}
                </Typography>
                <MaterialTable title={`Playlist #${selectedPlaylist}`}
                               columns={[{title: "Song", field: "songName"}, {title: "Artists", field: "songArtists"}]}
                               data={appState.data[selectedPlaylist].song_names.map((songName, idx) => {
                                   return {
                                       songName: songName,
                                       songArtists: appState.data[selectedPlaylist].artists[idx].join(", ")
                                   };
                               })
                               }/>
                <div style={{paddingTop: "50px"}}>
                    <Grid container justify="center" direction="row" spacing={10}>
                        {appState.data.map((entry, idx) => (
                            <Button variant="contained" color="primary" key={idx} onClick={() => {
                                setSelectedPlaylist(idx)
                            }}>{idx}</Button>))}
                    </Grid>
                </div>
            < /Grid>;
    }

    return (
        <
            div
            className="App">
            < header
                className="App-header">
                {componentToRender}
            < /header>
        </div>
    );
}

export default App;
