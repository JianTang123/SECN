
import torch.nn as nn
import torch


class SECN(nn.Module):
    def __init__(self, seg_size=16, in_c=1, num_classes=10, conv_kernel_size1=3, conv_kernel_size2=9,
                 conv_kernel_size3=15,
                 embedding_dim=96):
                 
        super(SECN, self).__init__()
        self.num_classes = num_classes
        self.Sequential_embedding = nn.Conv1d(in_c, embedding_dim, kernel_size=seg_size, stride=seg_size)
        


        self.conv_block0 = nn.Sequential(

            nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, conv_kernel_size1, groups=embedding_dim,
                          padding=int((conv_kernel_size1 - 1) / 2)),
                nn.GELU(),
               
                nn.BatchNorm1d(embedding_dim)
            ),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.GELU(),
            
            nn.BatchNorm1d(embedding_dim)

        )

        self.conv_block1 = nn.Sequential(

            nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, conv_kernel_size2, groups=embedding_dim,
                          padding=int((conv_kernel_size2 - 1) / 2)),
                nn.GELU(),
                
                nn.BatchNorm1d(embedding_dim)
            ),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.GELU(),
            
            nn.BatchNorm1d(embedding_dim)

        )
        self.conv_block2= nn.Sequential(

            nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, conv_kernel_size3, groups=embedding_dim,
                          padding=int((conv_kernel_size3 - 1) / 2)),
                nn.GELU(),
                
                nn.BatchNorm1d(embedding_dim)
            ),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.GELU(),
            
            nn.BatchNorm1d(embedding_dim)

        )

        

        self.head = nn.Linear(embedding_dim*3, num_classes)
        


    def forward(self, x):
        

        x = self.Sequential_embedding(x)
        
        y1 = self.conv_block0(x)
        
        y2 = self.conv_block1(x)
  
        y3 = self.conv_block2(x)


        y = torch.cat((y1, y2, y3), 1)
        y = nn.AdaptiveAvgPool1d(1)(y)
        y = y.view(y.size(0), -1)
        y = self.head(y)

        return y
