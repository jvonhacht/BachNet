var input = {
    'overfit': false,
    'melody': [],
};

const synth = new Tone.Synth().toMaster();

function add_note(note) {
    switch(note) {
        case 'C':
            input['melody'].push(72)   
            play_note(note)           
            break;
        case 'C#':
            input['melody'].push(73)  
            play_note(note)            
            break;
        case 'D':
            input['melody'].push(74)    
            play_note(note)           
            break;
        case 'D#':
            input['melody'].push(75)               
            play_note(note)
            break;
        case 'E':
            input['melody'].push(76)               
            play_note(note)
            break;
        case 'F':
            input['melody'].push(77)               
            play_note(note)
            break;
        case 'F#':
            input['melody'].push(78)               
            play_note(note)
            break;
        case 'G':
            input['melody'].push(79)               
            play_note(note)
            break;
        case 'G#':
            input['melody'].push(80)               
            play_note(note)
            break;
        case 'A':
            input['melody'].push(81)               
            play_note(note)
            break;
        case 'A#':
            input['melody'].push(82)  
            play_note(note)             
            break;
        case 'B':
            input['melody'].push(83)               
            play_note(note)
            break;                    
    }
    console.log(input['melody'])
}

function play_note(note) {
    synth.triggerAttackRelease(note + '5', "8n");          
}