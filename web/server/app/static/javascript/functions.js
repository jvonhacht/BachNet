var input = {
    'overfit': false,
    'melody': [],
};

// Initialise synth
const synth = new Tone.FMSynth();
synth.oscillator.type = "sine";
synth.toMaster();

var current_note = "";
  
/**
 * Adds note to model input
 */
function add_note() {    
    switch(current_note) {
        case 'C':
            input['melody'].push(72)               
            break;
        case 'C#':
            input['melody'].push(73)              
            break;
        case 'D':
            input['melody'].push(74)                
            break;
        case 'D#':
            input['melody'].push(75)               
            break;
        case 'E':
            input['melody'].push(76)               
            break;
        case 'F':
            input['melody'].push(77)                           
            break;
        case 'F#':
            input['melody'].push(78)                           
            break;
        case 'G':
            input['melody'].push(79)               
            break;
        case 'G#':
            input['melody'].push(80)               
            break;
        case 'A':
            input['melody'].push(81)                           
            break;
        case 'A#':
            input['melody'].push(82)              
            break;
        case 'B':
            input['melody'].push(83)                           
            break; 
        default:
            console.log("select note to add");
            break;                  
    }    
    console.log(input['melody']);
}

/**
 * Play note and add it to input
 * @param {string} note to add
 */
function play_note(note) {
    current_note = note;
    synth.triggerAttackRelease(note + '5', "8n");          
}

/**
 * Remove the last added note
 */
function remove_note() {
   input['melody'].pop();
   console.log(input['melody']);
}

/**
 * Sends a post request that generates 
 * harmonies from the inserted notes
 */
function harmonize() { 
    // create post form
    const form = document.createElement('form');
    form.method = 'post';
    form.action = '/harmonize';

    // create post body
    field = document.createElement('input');  
    field.name = 'input';    
    field.value = input;       
    field.type = 'hidden';  
    form.appendChild(field);      
    document.body.appendChild(form);

    // send post request
    form.submit();
  }