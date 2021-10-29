import torch
import torch.nn as nn


class model_encdec(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a attention layer with BiGRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    """
    def __init__(self, settings):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.att_size = settings['att_size']

        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # temporal encoding
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        
        # encoder-decoder
        
        # encoder
        self.encoder_past = nn.GRU(input_gru, self.dim_embedding_key,1, batch_first=True)
        self.encoder_fut = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        
        # print("encoder")
        # print(self.encoder_fut,self.encoder_fut)
        # # Attention
        
        # # self.attention = Attention(2* self.dim_embedding_key, 2* self.dim_embedding_key, att_size)  # attention network
        # self.attn_en = nn.Linear(self.dim_embedding_key + self.dim_embedding_key, self.att_size)
        # self.attn_de = nn.Linear(2* self.dim_embedding_key + self.dim_embedding_key, self.att_size)
        # self.attn_size = nn.Linear(self.att_size, 1)
        
        # decoder
        # self.decoder = nn.GRU( 2*self.dim_embedding_key , 2*self.dim_embedding_key ) #Input batch size 32 doesn't match hidden0 batch size 1
        # print("decoder")
        # print(self.decoder)
        
        self.decoder = nn.GRU(self.dim_embedding_key * 2, self.dim_embedding_key * 2, 1, batch_first=False)

        self.attn1 = nn.Linear(self.dim_embedding_key + self.dim_embedding_key, self.att_size)
        self.attn2 = nn.Linear(self.att_size, 1)
        
        # print("attn1")
        # print(self.attn1)
        # print("attn2")
        # print(self.attn2)
        
        self.FC_output = torch.nn.Linear(self.dim_embedding_key * 2 , 2)
        # self.op_traj = nn.Linear(self.dim_embedding_key, 2)

        
        # self.init_h = nn.Linear(self.dim_embedding_key * 2, self.dim_embedding_key * 2)
        # self.fcs = nn.Linear(self.dim_embedding_key * 2, self.dim_embedding_key * 2)
        # self.sigmoid = nn.Sigmoid()
 
    
 
    # # encoder-decoder
    #     self.encoder_past = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
    #     self.encoder_fut = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
    #     self.decoder = nn.GRU(self.dim_embedding_key * 2, self.dim_embedding_key * 2, 1, batch_first=False)
    #     self.FC_output = torch.nn.Linear(self.dim_embedding_key * 2, 2)
         
        # # Decoder:
        # self.dec_gru = nn.GRUCell(2*self.waypt_enc_size, self.traj_enc_size)
        # self.attn1 = nn.Linear(2*self.waypt_enc_size + self.traj_enc_size, self.att_size)
        # self.attn2 = nn.Linear(self.att_size, 1)
        # self.op_traj = nn.Linear(self.traj_enc_size, 2)
        
       
        
        

    

        # activation function
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax_att = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

        # weight initialization: kaiming
        self.reset_parameters()
        
        
    # def init_hidden_state(self, decoder):
    #     h = self.init_h(decoder.mean(dim=1))  # (batch_size, decoder_dim)
    #     return h
        

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.conv_fut.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_hh_l0)
        # nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        # nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.attn1.weight)
        nn.init.kaiming_normal_(self.attn2.weight)
        nn.init.kaiming_normal_(self.FC_output.weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.conv_fut.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.encoder_fut.bias_ih_l0)
        nn.init.zeros_(self.encoder_fut.bias_hh_l0)
        # nn.init.zeros_(self.decoder.bias_ih_l0)
        # nn.init.zeros_(self.decoder.bias_hh_l0)        
        nn.init.zeros_(self.attn1.bias)
        nn.init.zeros_(self.attn2.bias)
        nn.init.zeros_(self.FC_output.bias)
        


    def forward(self, past, future):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """
        
        
        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key *2) # dim , row , col
        # print("zero_padding")
        # print(zero_padding.shape)
        
        prediction = torch.Tensor()
        present = past[:, -1, :2].unsqueeze(1)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()
            prediction = prediction.cuda()

        # temporal encoding for past
        past = torch.transpose(past, 1, 2)
        past_embed = self.leaky_relu(self.conv_past(past))
        past_embed = torch.transpose(past_embed, 1, 2)

        # temporal encoding for future
        future = torch.transpose(future, 1, 2)
        future_embed = self.leaky_relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)

        # sequence encoding
        output_past, state_past = self.encoder_past(past_embed)
        output_fut, state_fut = self.encoder_fut(future_embed)
        
        
        # state concatenation and decoding
        state_conc = torch.cat((state_past, state_fut), 2)
        h = state_fut.squeeze()
        h2 = state_past.squeeze()
        input_fut = state_conc
        state_fut = zero_padding
        # print("state_conc")
        # print(state_conc.shape)
        # print("output_past")
        # print(output_past.shape)
        # print("output_fut")
        # print(output_fut.shape)
        
        for i in range(self.future_len):
                                     
            att_wts = self.softmax_att(self.attn2(self.tanh(self.attn1(torch.cat(  (h2.repeat(h2.shape[0], 1, 1),
                                                                                     h.repeat(h.shape[0], 1, 1) )  , 2)))))           
            # print("att_wts.shape")
            # print(att_wts.shape)
            # print("h.shape")
            # print(h.shape)
            # print("state_fut.shape")
            # print(state_fut.shape)
            # print("h2.shape")
            # print(h2.shape)
            # print("state_past.shape")
            # print(state_past.shape)

            ip = att_wts.repeat(1, 1, state_conc.shape[2])*state_conc

            ip = ip.unsqueeze(1)

            ip = ip.sum(dim=0)
            # print("ip.shape")
            # print(ip.shape)
            # print("state_fut.shape")
            # print(state_fut.shape)
            
            output_decoder, state_fut = self.decoder(ip, state_fut) #Input batch size 32 doesn't match hidden0 batch size 0
            # print("After  output_decoder")
            displacement_next = self.FC_output(output_decoder)
            # print("After FC Layer displacement_next")
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_fut = zero_padding
            
        return prediction
    





        
        # # Attention
        # num_pixels = (self.dim_embedding_key*2).size(1)
        # encoder_out = torch.Tensor(self.batch_size, num_pixels, self.dim_embedding_key*2) #encoded images, a tensor of dimension 
        # decoder_hidden = torch.Tensor(self.batch_size, self.dim_embedding_key*2) # previous decoder output, a tensor of dimension 
        
        
        # att1 = self.attn_en(encoder_out) # (batch_size, num_pixels, attention_dim)
        # att2 = self.attn_de(decoder_hidden)  # (batch_size, attention_dim)
        # att3 = self.attn_size(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        # soft = nn.Softmax(att3)
        # att_w_en = (encoder_out * soft.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        
