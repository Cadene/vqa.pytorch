$(document).ready(function () {

// Image

$(document).on('change', '.btn-file :file', function() {
    var input = $(this),
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [label]);
    });

    $('.btn-file :file').on('fileselect', function(event, label) {
        
        var input = $(this).parents('.input-group').find(':text'),
            log = label;
        
        if( input.length ) {
            input.val(log);
        } else {
            if( log ) alert(log);
        }
    
    });
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            
            reader.onload = function (e) {
                $('#vqa-visual').attr('src', e.target.result);
            }
            
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imgInp").change(function(){
        readURL(this);
    });     

// Send Image + Question

var formBasic = function () {
    var formData = $("#formBasic").serialize();
    var data = { visual : $('#vqa-visual').attr('src'),
                 question : $('#vqa-question').val()}
    console.log(data);
    $.ajax({

        type: 'post',
        data: data,
        dataType: 'json',
        url: 'http://', // your global ip address and port

        // error: function () {
        //     alert("There was an error processing this page.");
        //     return false;
        // },

        complete: function (output) {
            //console.log(output);
            //console.log(output.responseText);
            var ul = $('<ul></ul>');
            for (i=0; i < output.responseJSON.ans.length; i++)
            { 
                var li = $('<li></li>');
                var span = $('<span></span>');

                span.text(output.responseJSON.ans[i]+' ('+output.responseJSON.val[i]+')');

                li.append(span);
                ul.append(li);
            }

            for (i=0; i < output.responseJSON.att.length; i++)
            {
                var img = $('<img src="'+output.responseJSON.att[i]+'"/>');

                ul.append(img);
            }

            $('#vqa-answer').append(ul);
        }
    });
    return false;
};

   $("#basic-submit").on("click", function (e) {
   e.preventDefault();
   formBasic();
});
});