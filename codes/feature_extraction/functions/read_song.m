function [song_title, gt] = read_song(songdata_path,song_ind)

fname = songdata_path;
songdata = readtable(fname);

song_title = songdata.ConcertName(song_ind);
song_title = song_title{1,1};
gt=songdata.Boundaries(song_ind);
gt=str2num(gt{1,1});
