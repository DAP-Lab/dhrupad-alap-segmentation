function [song_title, gt] = read_song(song_ind)

fname = '../../annotations/train_data.csv';
songdata = readtable(fname);

song_title = songdata.ConcertName(song_ind);
song_title = song_title{1,1};
gt=songdata.Boundaries(song_ind);
gt=str2num(gt{1,1});
